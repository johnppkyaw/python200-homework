from dotenv import load_dotenv
from openai import OpenAI
import json

#Task 1: Setup and System Prompt
load_dotenv()
client = OpenAI()

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400,
    )
    return response.choices[0].message.content

system_prompt = """
  You are an expert in job application coaching and specialized in helping the career changers refine their resume and cover letter.  Below are the requirements:
  1) You should stay focused on the job application materials.  
  2) You should always remind the user to review and edit your output before submitting anywhere.  
  3) You should let the user know that you may not know the user's specific industry norms, and that the user should use their own judgment.  
  4) You should be professional and encouraging.
"""

#Q: Add a comment explaining at least one deliberate choice you made in writing the system prompt and why.

#A: I specifically added "career changers" as the target audience since many of us are in the similar situation and would be helpful for AI to focus on this type of job seekers, rather than everyone.

#Task 2: Bullet Point Rewriter
def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)
    print(bullet_text)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Respond ONLY with a valid JSON, no other text.  Return ONLY a valid JSON list of objects. Do not include markdown formatting 
    like ```json.  Each object must have "original" and "improved" keys.

    Bullet points:
    {bullet_text}
    """

    messages = [{"role": "user", "content": prompt}]
    try:
      response_json = get_completion(messages)
      data = json.loads(response_json)
      for each_bullet in data:
          print (f'The original: {each_bullet['original']}')
          print (f'The improved: {each_bullet['improved']}')

    except json.JSONDecodeError as e:
      print("Invalid JSON syntax:", e)
      print("raw response: \n", response_json)

bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

rewrite_bullets(bullets)

#Q: What makes these bullets weak, and what kinds of changes did the model suggest?
#A: These bullets were weak due to the vague information without any specific context.  The mode suggested by adding more details such as the what, the how and the how much.

#Task 3: Cover Letter Generator
def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Python Developer at a healthcare startup
    Background: Two years as a clinical auditor; self-taught in SQL; currently completing advanced Python, AI, and Machine Learning coursework at Code the Dream.
    Opening: In my work as a clinical auditor, I realized that my favorite part of the day wasn't the review itself, but finding ways to organize the data more effectively. This led me to a deep dive into Python and Machine Learning through Code the Dream, where I’ve spent the last several months building predictive models and advanced data pipelines. I am excited to join [Company] to apply my clinical domain knowledge and my technical Python skills to your current infrastructure.

    Example: The "Medical Accuracy to Full-Stack" Pivot
    Role: Junior Full Stack Developer at an insurance company
    Background: I spent years in the medical field as a scribe and auditor, which basically meant I was the last line of defense for accuracy in patient charts. After training others on how to document correctly, I realized I wanted to be the one building the systems that make that documentation easier, rather than just checking it for errors. Through Code the Dream, I’ve moved from analyzing medical reports to building full-stack applications with React and Node.js. I’m applying to [Company] because I’m looking for a team where my experience in high-pressure environments and my new technical skills in JavaScript can help build tools that actually work for the people using them.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    # Your code here: call get_completion() and return the result
    result = get_completion(messages)
    return result

job_title = "Junior Data Engineer"
background = "Five years of experience as a middle school math teacher; recently completed \
a Python course and built data pipelines using Prefect and Pandas."

print(generate_cover_letter(job_title, background))

#Q: Why did you choose those particular examples? 
#A: They are from my personal background.  They fit my style and make the letter like I personally wrote it, instead of making the AI guess it.

#Q: What does the few-shot pattern help control in the output?
#A: It keeps that model from drifting off from the desired pattern, voice and logic of the response.

#Task 4: Moderation Check
def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
    if flagged:
        print("Please rephrase your prompt.  It was inappropriate.")
        return False
    else:
        return True

print(f"Flagged: ", is_safe("shut your mouth"))
print(f"Safe: ", is_safe("hello"))

#Task 5: The Chatbot Loop
def run_chatbot():
    # 1. Initialize conversation history with my system prompt from above.
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        #    (hint: look for keywords like "bullet" or "resume" in user_input.lower())
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            rewrite_bullets(raw_bullets)

        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            result = generate_cover_letter(job_title, background)
            print(result)

        # 7. Otherwise, handle it as a regular chat turn
        else:
            message = {
                "role": "user",
                "content": user_input
            }
            messages.append(message)
            response = get_completion(messages)
            print(response)
            assistant_message = {
                "role": "assistant",
                "content": response
            }
            messages.append(assistant_message)


if __name__ == "__main__":
    run_chatbot()

#Task 6: Ethics Reflection
#Option A — Comment block

#Q: Your bot was trained on text written by and about certain kinds of people. How might this produce biased advice? Could it favor certain communication styles, industries, or cultural backgrounds?

#A: The bot's advice is biased towards the jobs in the American culture.  So, it favors towards the communication style that is backed by evidence rather than vague descriptions.  Therefore, anyone who is already familiar with the culture and the communication style is more advantageous than others in getting the interviews.  Since the bot was trained specifically for the people in tech or finance industry, it will not be helpful for the job seekers who are in a different career.

#Q: What could go wrong if a job-seeker submitted the bot's output directly — without reviewing it — to a real employer?

#A: The bot's output will not always be accurate, even with the instructions.  It can make up the skill sets and work experience that the job seeker might not have which will create a difficult situation during the interview.  Also, its writing style may not sound human-like and may sound generic that it can be easily recognizable by the hiring mangers as a letter created by AI. With these risks, using the output without revising will decrease the chance getting the interviews.
