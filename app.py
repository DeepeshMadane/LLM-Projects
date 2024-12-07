from flask import Flask, render_template, request
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

app = Flask(__name__)

# Initialize the language model
model = CTransformers(
    model="TheBloke/Llama-2-7B-Chat-GGML",
    model_type="llama",
    config={"max_new_tokens": 256, "temperature": 0.7}
)

# Function to generate a random question
def create_question(category):
    question_prompt = """
    Create a simple, easy-to-understand question specifically about {category}.
    Ensure the question is directly related to {category} and is suitable for a general knowledge quiz.
    """
    prompt = PromptTemplate(input_variables=["category"], template=question_prompt)
    response = model.invoke(prompt.format(category=category)).strip()
    return response.split("\n")[0].strip()

# Function to fetch the correct answer
def fetch_answer(question_text):
    answer_prompt = """
    Question: {question_text}
    
    Provide a concise and accurate answer to the question.
    """
    prompt = PromptTemplate(input_variables=["question_text"], template=answer_prompt)
    response = model.invoke(prompt.format(question_text=question_text)).strip()
    return response

@app.route("/", methods=["GET", "POST"])
def index():
    question = None
    user_answer = None
    feedback = None
    category = "Sports"  # Default category

    if request.method == "POST":
        category = request.form.get("category", "Sports") 
        user_answer = request.form.get("user_answer")

        if "generate" in request.form:
            question = create_question(category)
        elif "validate" in request.form and request.form.get("question"):
            question = request.form.get("question")
            correct_answer = fetch_answer(question)
            if correct_answer.lower() == user_answer.lower():
                feedback = "Correct! Well done!"
            else:
                feedback = f"Incorrect. The correct answer is: {correct_answer}"

    return render_template("index.html", question=question, feedback=feedback, user_answer=user_answer, category=category)

if __name__ == "__main__":
    app.run(debug=True)
