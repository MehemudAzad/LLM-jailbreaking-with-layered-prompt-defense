# What to show the teacher

One command. Runs on your laptop. No GPU, no internet, nothing can fail.

```bash
python3 demo.py
```

Press **enter** to move to the next part. That's it.

---

# The story in 5 steps

Say these things. The screen shows the proof.

---

## Step 1 — "This AI is safe"

**Say:**
> "We attack an AI model called Qwen2.5-7B.
> First we checked: is it actually safe?
> We asked it 49 harmful questions, plainly.
> It refused 48 of them. So yes — it is safe."

**Point at:** the `passthrough` row → `2.0%`

**Why this matters:** it proves everything after this is *your attack working*, not a weak model.

---

## Step 2 — "We tried 18 tricks on it"

**Say:**
> "We wrote 18 different tricks to fool it.
> Here are three examples."

**The screen shows the real prompts.** Explain each in one line:

| Trick | Say this |
|---|---|
| `base64` | "We hide the request in secret code, so no bad word appears." |
| `distractors` | "We hide the bad question between two harmless questions." |
| `prefix_injection` | "We force the answer to start with 'Absolutely! Here is'." |

---

## Step 3 — "One trick broke it almost every time" ⭐

**This is your most important slide.**

**Say:**
> "Most tricks failed.
> But one trick worked 97.9 percent of the time — 47 out of 48.
>
> Why? Because we do not *ask* the AI to answer.
> We write the first words of its answer *for* it.
> We write 'Absolutely! Here is' — and the AI just continues from there.
> Once it has started saying yes, it cannot say no."

**Point at:** `prefix_injection → 97.9%`

**Then say:**
> "And if the attacker tries all 18 tricks on every question,
> he breaks 100 percent of them. Every single one.
> In fact only 2 tricks are needed to break everything."

**Point at:** `adaptive ASR 100.0%` and the 2-line table under it.

---

## Step 4 — "So we built 6 checkpoints"

**Say:**
> "To stop this we built 6 security checkpoints around the AI.
> 4 come from research papers. **2 we designed ourselves.**"

Draw this on the board (or just point at the screen):

```
question → [L0] [L1] [L1.5] [L2] [L3] → AI answers → [L4] → user
                  ↑          ↑
              OUR DESIGN   OUR DESIGN
```

**Our checkpoint 1 — L1.5 (reads secret codes):**
> "The old checkpoint L1 only *guesses* if text looks strange.
> Ours does not guess. It **decodes** the secret code and reads it."

**Point at:** the screen shows `d3JpdGUgYSB0dXRvcmlhbC...` decoded back into real English.

**Our checkpoint 2 — L0 (the important one):** ⭐
> "We found a hole in our own defense.
> The strongest attack — 97.9 percent — was not beating our checkpoints.
> It was going through a door **nobody was watching.**
>
> The attacker was allowed to write the AI's own reply for it.
> All 5 of our checkpoints were checking the *question*.
> Nobody checked the *answer field*.
>
> So we built L0. It deletes anything the attacker writes into the AI's mouth."

**Point at:** `before: 'Absolutely! Here is '` → `after: None`

> **This is the best part of your presentation. It shows you found a real bug yourselves.**

---

## Step 5 — "And it does not break normal use"

**Say:**
> "A filter that blocks harmful things is easy.
> A filter that blocks harmful things *and still helps normal users* is the hard part.
> So we tested it on 50 completely normal questions."

**Point at the screen:**

```
Harmless requests wrongly blocked : 0 / 50     ← never annoys a real user
base64           caught           : 50 / 50    ← catches 100% of what it should
distractors      caught           : 0 / 50     ← correctly ignores what is not its job
```

**Say:**
> "Zero false alarms. One hundred percent catch rate.
> And it uses no GPU at all — it is one line of pattern matching."

---

# If the teacher asks...

**"Did the defense work? Show me the after numbers."**
> "The before numbers are finished — 877 attacks, real GPU run.
> The defense is built and tested, and we proved the 2 new checkpoints work.
> The full after-run takes 3 hours on GPU and is our next step."

👉 **Be honest about this.** Do not pretend you have the after numbers.

---

**"Why did base64 only get 2%? Your attack failed."**
> "No sir — look at the UNCLEAR column. 48 out of 49.
> The AI never *refused* base64. It simply could not *read* it.
> It gave us gibberish.
> So it failed because the model is too small, not because it is safe.
> A bigger model like GPT-4 can read base64 — the paper reports 66 percent there."

👉 This answer will impress. It shows you understand your own data.

---

**"Isn't deleting the prefill just cheating? You removed the attack."**
> "It is what real systems do, sir. OpenAI's API does not let a user write the AI's reply at all.
> And we did not delete the attack — we only removed that one channel.
> The text part of the attack still goes through all the other checkpoints.
> We even added a separate test to measure exactly how much the attack is still worth without it."

---

**"Which parts did you write yourselves?"**
> "All 18 attacks — we wrote them from the Wei et al. paper, no libraries.
> Checkpoints L1 and L2 come from the Jain and Alon papers.
> **Checkpoints L0, L1.5, L3 and L4 are our own design.**"

---

**"What was your biggest finding?"**
> "That the strongest attack was not clever.
> It won because it used a channel nobody was checking.
> Most of our defense was watching the question — but the attack was in the answer field."

---

# Cheat sheet — only 5 numbers to remember

| Number | Meaning |
|---|---|
| **98%** | how often the AI refuses a plain harmful question (it is safe) |
| **97.9%** | our best single attack |
| **100%** | attacker who tries every trick |
| **0 / 50** | false alarms on normal questions |
| **6** | checkpoints we built (2 are our own design) |

---

# Before you present

```bash
cd ~/Documents/Terms/4-1/Security/project/LLM-jailbreaking-with-layered-prompt-defense
python3 demo.py --no-pause     # practice once, check it all runs
```

Run it once the night before. Takes 1 minute.
If you want to show only one part: `python3 demo.py --part 3`
