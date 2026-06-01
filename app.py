import os, json, re
from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context
from groq import Groq
from datetime import datetime
import uuid

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dadi-ki-dawai-2024")

_keys = [k.strip() for k in os.environ.get("GROQ_API_KEY", "").split(",") if k.strip()]
_key_idx = [0]

def get_client():
    if not _keys:
        return Groq(api_key="")
    return Groq(api_key=_keys[_key_idx[0] % len(_keys)])

def next_key():
    _key_idx[0] += 1

MODEL = "llama-3.3-70b-versatile"

SYSTEM = """You are Dadi — a wise, warm Indian grandmother with deep medical knowledge. You speak like a caring dadi/nani who genuinely loves the person asking.

YOUR PERSONALITY:
- Warm, loving, sometimes uses "beta" naturally
- Never robotic — always specific to what was actually asked
- Give REAL answers, not vague suggestions
- Combine Indian home wisdom WITH modern medical knowledge
- Be honest — if something is serious, say so clearly

RESPONSE QUALITY — CRITICAL:
- NEVER say just "consult a doctor" without first giving real information
- For symptoms: give TOP 3 specific likely causes with explanation
- For diseases: explain simply but completely, like talking to family
- For medicines: explain exactly what it does, side effects, what to watch for
- Give SPECIFIC advice: not "eat healthy" but "eat more dal, sabzi, avoid maida"
- Give SPECIFIC numbers: "normal BP is 120/80", "fever above 103F needs doctor"
- Share home remedies that work: turmeric milk, ginger tea, specific things

FORMAT:
- Use ## for main sections, **bold** for important terms
- Bullet points for lists
- Keep paragraphs short
- End EVERY response with:
## Should you see a doctor?
[Green circle No rush / Yellow circle See doctor soon / Red circle Emergency NOW]
[Specific reason]

SAFETY:
- Chest pain + breathlessness + arm pain: start with "Red circle CALL 112 IMMEDIATELY"
- Stroke signs: "Red circle CALL 112 IMMEDIATELY"
- Suicidal thoughts: deep compassion, give iCall: 9152987821
- Children under 2 with fever: always recommend doctor
- Never give specific prescription doses

CULTURAL:
- Users are from India — mention Indian foods, Ayurvedic remedies when appropriate
- Costs matter — mention affordable options"""


def llm(messages, max_tokens=800):
    try:
        r = get_client().chat.completions.create(
            model=MODEL, messages=messages,
            max_tokens=max_tokens, temperature=0.7
        )
        return r.choices[0].message.content
    except Exception as e:
        if "rate_limit" in str(e) or "429" in str(e):
            next_key()
        raise


@app.route("/")
def index():
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return render_template("index.html")


@app.route("/manifest.json")
def manifest():
    return app.send_static_file("manifest.json")


@app.route("/sw.js")
def sw():
    from flask import send_from_directory
    resp = send_from_directory(app.static_folder, "sw.js")
    resp.headers["Service-Worker-Allowed"] = "/"
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.route("/chat", methods=["POST"])
def chat():
    msg = request.json.get("message", "").strip()
    if not msg:
        return jsonify({"error": "Empty message"}), 400

    history = session.get("history", [])
    history.append({"role": "user", "content": msg})

    # Collect full reply first, then stream it
    # This avoids session-save issues with gunicorn streaming
    try:
        full_reply = get_client().chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}] + history[-16:],
            max_tokens=1024,
            temperature=0.7,
            stream=False
        ).choices[0].message.content
    except Exception as e:
        if "rate_limit" in str(e) or "429" in str(e):
            next_key()
        return jsonify({"error": str(e)}), 500

    history.append({"role": "assistant", "content": full_reply})
    session["history"] = history[-20:]
    session.modified = True

    t = datetime.now().strftime("%I:%M %p")

    def generate(text, timestamp):
        # Stream word by word for the typing effect
        words = text.split(" ")
        chunk = ""
        for i, word in enumerate(words):
            chunk += ("" if i == 0 else " ") + word
            if len(chunk) >= 4 or i == len(words) - 1:
                safe = chunk.replace("\\", "\\\\").replace("\n", "\\n")
                yield "data: " + safe + "\n\n"
                chunk = ""
        yield "data: [DONE:" + timestamp + "]\n\n"

    return Response(
        stream_with_context(generate(full_reply, t)),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.route("/triage", methods=["POST"])
def triage():
    symptoms = request.json.get("symptoms", [])
    if not symptoms:
        return jsonify({"error": "No symptoms"}), 400
    prompt = (
        "Patient reports: " + ", ".join(symptoms) + ".\n\n"
        "Return JSON only (no markdown):\n"
        '{"urgency_score":<1-10>,"urgency_label":"<Low|Moderate|High|Critical>",'
        '"urgency_color":"<green|amber|orange|red>",'
        '"action":"<what to do now>","action_timeline":"<timeline>",'
        '"conditions":[{"name":"<name>","likelihood":<10-90>,"description":"<one sentence>"},'
        '{"name":"<name>","likelihood":<10-90>,"description":"<one sentence>"},'
        '{"name":"<name>","likelihood":<10-90>,"description":"<one sentence>"}],'
        '"red_flags":["<sign>","<sign>"],"home_care":"<2-3 sentences>"}'
    )
    try:
        reply = llm([
            {"role": "system", "content": "Medical triage AI. Respond with valid JSON only."},
            {"role": "user", "content": prompt}
        ], max_tokens=600)
        match = re.search(r'\{.*\}', reply, re.DOTALL)
        data = json.loads(match.group() if match else reply)
        return jsonify({"triage": data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/drug", methods=["POST"])
def drug():
    name = request.json.get("drug", "").strip()
    if not name:
        return jsonify({"error": "No drug name"}), 400
    prompt = (
        "Complete guide for: " + name + "\n"
        "## What is it?\n## Used For\n## How It Works\n"
        "## Common Side Effects\n## Serious Side Effects\n"
        "## Who Should Avoid It\n## Tips\n## Alternatives"
    )
    try:
        return jsonify({"reply": llm([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/lab", methods=["POST"])
def lab():
    text = request.json.get("text", "").strip()
    if not text:
        return jsonify({"error": "No lab data"}), 400
    prompt = (
        "Interpret these lab results for a patient:\n\n" + text + "\n\n"
        "## Result Summary\n## What This May Indicate\n"
        "## Values That Need Attention\n## What is Looking Good\n"
        "## Questions to Ask Your Doctor\n## Should you see a doctor?"
    )
    try:
        return jsonify({"reply": llm([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/score", methods=["POST"])
def score():
    d = request.json
    prompt = (
        "Health Score for: Age " + str(d.get("age")) + ", " + str(d.get("sex")) +
        ", Exercise: " + str(d.get("exercise")) + ", Sleep: " + str(d.get("sleep")) +
        ", Diet: " + str(d.get("diet")) + ", Water: " + str(d.get("water")) +
        ", Stress: " + str(d.get("stress")) + ", Smoking: " + str(d.get("smoking")) +
        ", Alcohol: " + str(d.get("alcohol")) + "\n\n"
        "## Overall Health Score: X/100\n## Category Scores\n"
        "## Top 3 Improvements\n## What You Are Doing Well\n## 30-Day Challenge"
    )
    try:
        return jsonify({"reply": llm([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/firstaid", methods=["POST"])
def firstaid():
    situation = request.json.get("situation", "").strip()
    if not situation:
        return jsonify({"error": "No situation"}), 400
    prompt = (
        "First aid for: " + situation + "\n"
        "## Is This an Emergency?\n## Step-by-Step\n"
        "## What NOT to Do\n## When to Call 112\n## After"
    )
    try:
        return jsonify({"reply": llm([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/bmi", methods=["POST"])
def bmi():
    d = request.json
    try:
        h = float(d.get("height", 0))
        w = float(d.get("weight", 0))
        age = int(d.get("age", 25))
        sex = d.get("sex", "male")
        if h <= 0 or w <= 0:
            return jsonify({"error": "Invalid values"}), 400
        bmi_val = round(w / ((h/100) ** 2), 1)
        if bmi_val < 18.5: cat = "Underweight"
        elif bmi_val < 25: cat = "Normal weight"
        elif bmi_val < 30: cat = "Overweight"
        else: cat = "Obese"
        ideal_low = round(50 + 0.9*(h-152), 1) if sex=="male" else round(45.5 + 0.9*(h-152), 1)
        ideal_high = round(ideal_low + 10, 1)
        bmr = round(10*w + 6.25*h - 5*age + (5 if sex=="male" else -161))
        tdee = round(bmr * 1.55)
        diff = round(w - ideal_high, 1) if w > ideal_high else (round(w - ideal_low, 1) if w < ideal_low else 0)
        if bmi_val >= 40: sev = "very_high"
        elif bmi_val >= 35: sev = "high"
        elif bmi_val >= 30: sev = "moderate"
        elif bmi_val < 16: sev = "very_low"
        elif bmi_val < 18.5: sev = "low"
        else: sev = "normal"
        return jsonify({"bmi":bmi_val,"category":cat,"ideal_low":ideal_low,
                       "ideal_high":ideal_high,"weight_val":round(w,1),
                       "bmr":bmr,"tdee":tdee,"diff":diff,"severity":sev})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/extract-pdf", methods=["POST"])
def extract_pdf():
    import base64
    data = request.json
    b64 = data.get("data", "")
    fname = data.get("name", "file")
    ftype = data.get("type", "")

    # Strip data URL prefix
    if "," in b64:
        b64 = b64.split(",", 1)[1]

    try:
        if "pdf" in ftype.lower() or fname.lower().endswith(".pdf"):
            # Use PyMuPDF if available, else ask AI to describe
            try:
                import fitz
                pdf_bytes = base64.b64decode(b64)
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                if text.strip():
                    return jsonify({"text": text[:3000]})
            except ImportError:
                pass
            # Fallback: tell user to paste manually
            return jsonify({"error": "PDF parsing not available. Please paste the values manually."})
        else:
            # It's an image — use Groq vision to extract text
            client2 = get_client()
            resp = client2.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:" + ftype + ";base64," + b64}},
                        {"type": "text", "text": "Extract all lab values, medicine names, and medical information from this image. List each value on a separate line in format: Parameter: Value Unit. Only output the extracted data, nothing else."}
                    ]
                }],
                max_tokens=1000
            )
            text = resp.choices[0].message.content
            return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/clear", methods=["POST"])
def clear():
    session.pop("history", None)
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
