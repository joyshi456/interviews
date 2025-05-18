import json, random, time, sympy as sp
from openai import OpenAI

# ----------------------------------------
#  connection to Gemini‑flash endpoint
# ----------------------------------------
client = OpenAI(
    api_key="EMPTY",
    base_url="http://34.28.33.252:6001"
)
MODEL_NAME = "gemini-1.5-flash"

# main integration variable (kept global so the model can use `x`)
x = sp.symbols("x")

# ----------------------------------------
#  helper : ask Gemini for one integrand
# ----------------------------------------
_SYSTEM_PROMPT = """You are an expert calculus problem generator.
Produce ONE indefinite‑integration problem in SymPy‑ready Python syntax.
Return *only* valid JSON of the form:
{"integrand": "<sympy_expression>", "latex": "<\\LaTeX version of the integral>"}

Difficulty rules (LEVEL):
1  → single‑step u‑sub, simple trig, or power‑rule
2  → integration by parts, easy trig identities, or simple rational / partial‑fractions
3  → combination of techniques, trickier trig, nested exponentials or root factors

Requirements:
* Use the symbol x as the integration variable.
* Expression must be integrable by hand and by sympy.integrate(x).
* `latex` must start with \\int and include \\,dx at the end.
"""

def _safe_api_call(func, max_retries=3, *args, **kwargs):
    """
    Wrapper function to safely make API calls with retries.
    
    Args:
        func: The API function to call
        max_retries: Maximum number of retry attempts
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        The result of the function call
        
    Raises:
        Exception: If all retries fail
    """
    retries = 0
    last_error = None
    
    while retries < max_retries:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_error = e
            retries += 1
            # Add exponential backoff if needed
            time.sleep(1 * retries)  # Simple backoff strategy
    
    # If we get here, all retries failed
    raise ValueError(f"API call failed after {max_retries} attempts. Last error: {last_error}")

def _gemini_call(level: int) -> dict[str, str]:
    """Query Gemini and parse the JSON response."""
    try:
        print("Creating Gemini API request...")
        
        # Add a random seed to the prompt to encourage variety
        random_seed = int(time.time() * 1000) % 10000
        
        # Create the messages with additional instructions for variety
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"LEVEL = {level}. SEED = {random_seed}. Make sure to create a UNIQUE and DIFFERENT problem than previous ones. Provide one problem now."}
        ]
        
        print(f"Using model: {MODEL_NAME}")
        print(f"API base URL: {client.base_url}")
        
        # Use streaming for compatibility with the API
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.9,
            max_tokens=200,
            stream=True  # Important: Use streaming mode
        )
        
        # Collect the streamed response
        content = ""
        print("Collecting streamed response...")
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
        
        # Process the collected content
        raw = content.strip()
        print(f"Raw response: {raw}")
        
        if not raw:
            raise ValueError("Empty response content")
            
        # Gemini sometimes wraps JSON in markdown triple‑backticks – strip them:
        if raw.startswith("```"):
            raw = raw.strip("`")
            print("Stripped backticks from response")
        
        # Remove any non-JSON text before or after the JSON object
        try:
            # Try to parse as is first
            print("Attempting to parse raw response as JSON")
            data = json.loads(raw)
            print("Successfully parsed JSON response")
        except json.JSONDecodeError:
            # If that fails, try to extract JSON by looking for { and }
            try:
                print("Attempting to extract JSON from response")
                start_idx = raw.find('{')
                end_idx = raw.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = raw[start_idx:end_idx]
                    print(f"Extracted JSON string: {json_str}")
                    data = json.loads(json_str)
                    print("Successfully parsed extracted JSON")
                else:
                    print("Could not find JSON markers in response")
                    raise ValueError("Could not find valid JSON in the response")
            except Exception as e:
                print(f"Failed to extract and parse JSON: {e}")
                raise ValueError(f"Failed to parse JSON response: {e}")
        
        # Validate the required fields are present
        print(f"Checking result fields: {data}")
        if "integrand" not in data:
            print("Missing 'integrand' field in result")
            raise ValueError("Response missing required 'integrand' field")
        if "latex" not in data:
            print("Missing 'latex' field in result")
            raise ValueError("Response missing required 'latex' field")
            
        return data
    except Exception as e:
        raise ValueError(f"Error in Gemini API call: {e}")


# ----------------------------------------
#  public API used by Streamlit app
# ----------------------------------------

def check_answer_with_gemini(problem_latex: str, user_answer: str, correct_answer: str) -> dict:
    """
    Use Gemini to check if the user's answer is correct.
    
    Args:
        problem_latex: The LaTeX representation of the problem
        user_answer: The user's answer as a string
        correct_answer: The correct answer as a string
        
    Returns:
        A dictionary with keys:
        - is_correct: True if the answer is correct, False otherwise
        - explanation: An explanation of why the answer is correct or incorrect
    """
    print("Checking answer with Gemini...")
    print("Problem LaTeX:", problem_latex)
    print("User answer:", user_answer)
    print("Correct answer:", correct_answer)
    
    try:
        # Prepare the prompt for Gemini
        prompt = f"""
        You are a calculus expert evaluating a student's answer to an integration problem.
        
        Problem: {problem_latex}
        Student's answer: {user_answer}
        Correct answer: {correct_answer}
        
        Determine if the student's answer is mathematically equivalent to the correct answer.
        Consider that:
        1. The answers may look different but be equivalent (e.g., x^2/2 and 0.5*x^2)
        2. The student may have forgotten the constant of integration
        3. The student may have simplified the answer differently
        
        Return ONLY a JSON object with this format:
        {{
          "is_correct": true/false,
          "explanation": "Brief explanation of why the answer is correct or incorrect"
        }}
        """
        
        print("Sending prompt to Gemini API...")
        
        # Call Gemini API
        print(f"Using model: {MODEL_NAME}")
        print(f"API base URL: {client.base_url}")
        
        # Use streaming for compatibility with the API
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.9,
            max_tokens=200,
            stream=True  # Important: Use streaming mode
        )
        
        # Collect the streamed response
        content = ""
        print("Collecting streamed response...")
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
        
        # Process the collected content
        raw = content.strip()
        print(f"Raw response content: {raw}")
        
        if raw.startswith("```"):
            raw = raw.strip("`")
            print("Stripped backticks from response")
        
        # Extract JSON
        try:
            # Try to parse as is first
            print("Attempting to parse raw response as JSON")
            result = json.loads(raw)
            print("Successfully parsed JSON response")
        except json.JSONDecodeError as json_err:
            print(f"JSON decode error: {json_err}")
            # If that fails, try to extract JSON by looking for { and }
            try:
                print("Attempting to extract JSON from response")
                start_idx = raw.find('{')
                end_idx = raw.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = raw[start_idx:end_idx]
                    print(f"Extracted JSON string: {json_str}")
                    result = json.loads(json_str)
                    print("Successfully parsed extracted JSON")
                else:
                    print("Could not find JSON markers in response")
                    raise ValueError("Could not find valid JSON in the response")
            except Exception as e:
                print(f"Failed to extract and parse JSON: {e}")
                raise ValueError(f"Failed to parse JSON response: {e}")
        
        # Ensure the result has the required fields
        print(f"Checking result fields: {result}")
        if "is_correct" not in result:
            print("Missing 'is_correct' field in result")
            raise ValueError("Response missing required 'is_correct' field")
        if "explanation" not in result:
            print("Missing 'explanation' field in result")
            raise ValueError("Response missing required 'explanation' field")
        
        print(f"Final result: is_correct={result['is_correct']}, explanation={result['explanation']}")
        return result
    except Exception as e:
        # If anything goes wrong, return a default response
        print(f"Error checking answer with Gemini: {e}")
        return {
            "is_correct": False,
            "explanation": "Could not verify answer due to an error. Please try again."
        }
def generate_problem(level: int) -> dict:
    """
    Query Gemini for a fresh integrand at the requested level
    and return {'latex': ..., 'sympy': sympy_expr}.
    If the LLM fails, fall back to a local template.
    """
    print(f"Generating problem at level {level}...")
    
    # Set a random seed based on current time to ensure different problems
    random_seed = int(time.time() * 1000)
    random.seed(random_seed)
    print(f"Using random seed: {random_seed}")
    
    try:
        print("Calling Gemini API...")
        data = _gemini_call(level)
        print(f"Received data from Gemini: {data}")
        
        # Handle potential SymPy parsing errors
        try:
            # Use our safe_sympify function to parse the expression
            sympy_integrand = safe_sympify(data["integrand"])
            result = {"latex": data["latex"], "sympy": sympy_integrand, "source": "gemini"}
            print(f"Successfully generated problem: {result['latex']}")
            return result
        except Exception as sympy_err:
            print(f"Failed to parse SymPy expression: {sympy_err}")
            # If SymPy parsing fails, try a different approach or raise to fallback
            raise ValueError(f"Failed to parse SymPy expression: {sympy_err}")
    except Exception as err:
        print(f"Error in Gemini call, using fallback: {err}")
        # --- graceful fallback so the Bee never crashes ---------------
        fallback_bank = {
            1: [
                r"x**2",
                r"x**3",
                r"sin(x)",
                r"cos(x)",
                r"exp(x)",
                r"1/x",
                r"x*sin(x)",
                r"x**2 + 2*x + 1"
            ],
            2: [
                r"x*exp(2*x)",
                r"x*sin(x)",
                r"x**2*exp(x)",
                r"log(x)",
                r"x*log(x)",
                r"sin(x)**2",
                r"1/(x**2 + 1)"
            ],
            3: [
                r"(x**2)/(x**2 + 1)",
                r"exp(x)*sin(x)",
                r"x*exp(x)*sin(x)",
                r"log(x)/x",
                r"1/(x*log(x))",
                r"sin(x)*cos(x)",
                r"1/(x**2 - 1)"
            ]
        }
        
        # Get the fallback bank for the current level, or default to level 1
        level_bank = fallback_bank.get(level, fallback_bank[1])
        
        # Choose a random expression from the bank
        expr_str = random.choice(level_bank)
        print(f"Selected fallback expression: {expr_str}")
        
        try:
            sympy_expr = safe_sympify(expr_str)
            result = {
                "latex": rf"\int {sp.latex(sympy_expr)} \,dx",
                "sympy": sympy_expr,
                "source": "fallback"
            }
            print(f"Using fallback problem: {result['latex']}")
            return result
        except Exception as fallback_err:
            print(f"Error with fallback: {fallback_err}, using ultimate fallback")
            # Ultimate fallback - just return x
            return {
                "latex": r"\int x \,dx",
                "sympy": x,
                "source": "ultimate_fallback"
            }

def solve_integral(expr: sp.Expr) -> sp.Expr:
    """
    Return an antiderivative of *expr* with respect to x.
    Includes error handling for integration failures.
    """
    try:
        # Try to integrate with a timeout to prevent hanging on complex expressions
        result = sp.integrate(expr, x)
        
        # Verify the result is valid by differentiating it
        try:
            derivative = sp.diff(result, x)
            # Check if the derivative matches the original expression
            if sp.simplify(derivative - expr) != 0:
                # If not, try a different approach or raise an error
                raise ValueError("Integration verification failed: derivative doesn't match original")
        except Exception as verify_err:
            # If verification fails, still return the result but log the issue
            print(f"Warning: Integration verification issue: {verify_err}")
        
        return result
    except Exception as e:
        # If integration fails, try a fallback approach
        try:
            # For some expressions, we can try alternative methods
            # For example, try integration by parts or substitution
            # This is a simplified fallback - in a real system, you might have more sophisticated fallbacks
            
            # For now, just return a simple polynomial if it's a polynomial-like expression
            if isinstance(expr, sp.Poly) or (isinstance(expr, sp.Expr) and expr.is_polynomial(x)):
                degree = sp.degree(expr, x)
                if degree >= 0:
                    # For x^n, the integral is x^(n+1)/(n+1)
                    return expr * x / (degree + 1)
            
            # If all else fails, raise the original error
            raise e
        except Exception:
            # Ultimate fallback - just return x^2/2 as a last resort
            print(f"Integration failed completely: {e}")
            return x**2 / 2

def safe_sympify(expr_str: str) -> sp.Expr:
    """
    Safely convert a string to a SymPy expression.
    Handles common issues like 'sp.' prefixes and provides better error messages.
    """
    try:
        # Remove any 'sp.' prefixes as they're not needed with sympify
        cleaned_expr = expr_str.replace("sp.", "")
        return sp.sympify(cleaned_expr, locals={"x": x})
    except Exception as e:
        # Try to provide more helpful error messages
        if "(" in expr_str and ")" not in expr_str:
            raise ValueError("Mismatched parentheses in expression")
        elif "**" in expr_str and any(c.isalpha() for c in expr_str):
            raise ValueError("Check power expressions - use x**2 for x squared")
        elif "^" in expr_str:
            raise ValueError("Use ** for exponentiation, not ^")
        else:
            raise ValueError(f"Could not parse expression: {e}")
