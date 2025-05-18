# app.py
import streamlit as st
import sympy as sp
import random
import time
import pandas as pd
import re
# Assuming problem_bank.py is in the same directory and contains:
# generate_problem(level), solve_integral(sympy_expr), x (sympy.Symbol('x')), safe_sympify(str_expr)
from problem_bank import generate_problem, solve_integral, x, safe_sympify

# ----------------- constants -----------------
TOTAL_SECONDS = 600  # 10 minutes
LEVEL_UP = {0: 1, 4: 2, 8: 3}  # point thresholds → difficulty level

# ----------------- helper functions -----------------
def reset_game():
    """
    Reset the game to its initial state.
    """
    state.prob = None
    state.level = 1
    state.points = 0
    state.start = None
    state.answered = 0
    state.correct = 0
    st.rerun()

def save_to_leaderboard():
    """
    Save the current player's score to the leaderboard.
    """
    if not state.player_name or state.points == 0:
        return

    new_entry = {
        "Player": state.player_name,
        "Score": state.points,
        "Level": state.level,
        "Correct": state.correct,
        "Total": state.answered,
        "Accuracy": f"{(state.correct / max(1, state.answered)) * 100:.1f}%",
        "Date": time.strftime("%Y-%m-%d %H:%M")
    }

    if "leaderboard" not in state:
        state.leaderboard = pd.DataFrame([new_entry])
    else:
        state.leaderboard = pd.concat([state.leaderboard, pd.DataFrame([new_entry])], ignore_index=True)
        state.leaderboard = state.leaderboard.sort_values(by="Score", ascending=False).reset_index(drop=True)

# ----------------- session state -----------------
state = st.session_state

# Initialize state variables if they don't exist
default_values = {
    "player_name": "",
    "prob": None,
    "level": 1,
    "points": 0,
    "start": None,
    "answered": 0,
    "correct": 0,
    "leaderboard": pd.DataFrame(columns=["Player", "Score", "Level", "Correct", "Total", "Accuracy", "Date"]),
    "tab": "Play"
}
for key, value in default_values.items():
    if key not in state:
        state[key] = value

# ----------------- app layout -----------------
st.title("🧠 Integration Bee: Math Challenge")
st.write("Master calculus integrals and compete for the top spot on the leaderboard!")

tabs = st.tabs(["Play", "Leaderboard"])

# ----------------- Play Tab -----------------
with tabs[0]:
    if not state.player_name:
        st.header("Enter Your Name")
        col1, col2 = st.columns([3, 1])
        with col1:
            player_name_input = st.text_input("Your Name:", key="name_input_field")
        with col2:
            st.write("") # Spacer
            st.write("") # Spacer
            if st.button("Start Game", key="start_game_button"):
                if player_name_input.strip():
                    state.player_name = player_name_input.strip()
                    st.rerun()
                else:
                    st.warning("Please enter a name to start.")
    else:
        with st.expander("ℹ️ How to Play & Tips"):
            st.markdown("""
            ### How to Play
            1. **Solve integration problems**: Each correct answer earns points based on difficulty.
            2. **Enter your answer**: Type your solution using SymPy syntax (e.g., `x**2/2` for ∫x dx). Include `+ C` if you like, but it's not strictly necessary for SymPy to check equivalence (though good practice!).
            3. **Score points**: Correct answers earn points based on the current difficulty level.
            4. **Level up**: As you score more points, problems get harder!

            ### SymPy Syntax Tips
            - Use `x**2` for $x^2$
            - Use `sin(x)`, `cos(x)`, `tan(x)` for trigonometric functions
            - Use `exp(x)` for $e^x$
            - Use `log(x)` for natural logarithm ($\ln(x)$)
            - Use `sqrt(x)` for $\sqrt{x}$
            - Use `pi` for $\pi$
            - For fractions like 1/3, use `S(1)/3` or `Rational(1,3)`. Example: `(S(1)/3)*x**3`
            - Constants of integration (like `C`) can be defined as `Symbol('C')`.

            ### Common Integration Patterns
            - $\int x^n dx = \frac{x^{n+1}}{n+1} + C$ (for $n \neq -1$)
            - $\int \sin(x) dx = -\cos(x) + C$
            - $\int \cos(x) dx = \sin(x) + C$
            - $\int e^x dx = e^x + C$
            - $\int \frac{1}{x} dx = \ln(|x|) + C$ (SymPy: `log(abs(x))`)
            """)

        if not state.start:
            st.header(f"Welcome, {state.player_name}!")
            if st.button("Start 10-Minute Challenge"):
                state.start = time.time()
                state.prob = None
                st.rerun()

        if state.start:
            elapsed = time.time() - state.start
            remaining = max(0, int(TOTAL_SECONDS - elapsed))

            mins, secs = divmod(remaining, 60)

            col_time, col_level, col_score = st.columns(3)
            col_time.markdown(f"### ⏳ Time: `{mins:02d}:{secs:02d}`")
            level_emoji = {1: "🟢", 2: "🟡", 3: "🔴"}
            current_emoji = level_emoji.get(state.level, "🟢")
            col_level.markdown(f"### Level: {current_emoji} {state.level}")
            col_score.markdown(f"### Score: {state.points}")

            if remaining == 0:
                st.success("⏰ Time's up! Great job!")
                st.markdown(f"**Final Score:** {state.points} points")
                st.markdown(f"**Problems Solved:** {state.correct}/{state.answered}")
                if state.points > 0:
                    save_to_leaderboard()
                    st.success("Your score has been added to the leaderboard!")
                if st.button("Play Again?"):
                    reset_game()
                st.stop()

            if state.prob is None:
                try:
                    state.prob = generate_problem(state.level)
                except Exception as gen_err:
                    st.error(f"Error generating problem: {gen_err}. Using a fallback.")
                    state.prob = {"latex": r"\int x \,dx", "sympy": x, "id": "fallback_x"}

            st.subheader("Solve this integral:")
            try:
                st.latex(state.prob["latex"])
            except Exception as latex_err:
                st.error(f"Error displaying problem LaTeX: {latex_err}")
                st.write(f"Problem: Integrate `{state.prob['sympy']}` with respect to x")


            answer_input = st.text_input("∫ = ", key=f"answer_input_{state.prob.get('id', 0)}")
            submit_button = st.button("Submit Answer")

            if submit_button:
                if not answer_input.strip():
                    st.error("⚠️ Please enter an answer before submitting.")
                else:
                    state.answered += 1
                    try:
                        correct_expr_sympy = solve_integral(state.prob["sympy"])
                        correct_latex_display = sp.latex(correct_expr_sympy) + " + C"

                        try:
                            processed_answer = answer_input

                            # Pre-process common syntax sugar before SymPy parsing
                            # Handle implicit multiplication: number then letter (e.g., 2x -> 2*x)
                            processed_answer = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', processed_answer)
                            # Handle implicit multiplication: closing parenthesis then letter or opening parenthesis (e.g., )x -> )*x or )( -> )*()
                            processed_answer = re.sub(r'\)([a-zA-Z(])', r'\)*\1', processed_answer) # CORRECTED
                            # Handle implicit multiplication: letter/number then opening parenthesis (e.g., x( -> x*()
                            processed_answer = re.sub(r'([a-zA-Z0-9])\(', r'\1*(', processed_answer)


                            # Allow user to define 'C' or 'c' as a constant
                            user_expr_sympy = safe_sympify(processed_answer, locals={'C': sp.Symbol('C'), 'c': sp.Symbol('C'), 'S':sp.S, 'Rational':sp.Rational})


                            if processed_answer != answer_input:
                                st.info(f"Note: Your input '{answer_input}' was interpreted as '{processed_answer}' for SymPy.")

                            # Check if the difference is a constant
                            # Remove any 'C' from user_expr if they added it, as we compare antiderivatives
                            # A simple way is to substitute C with 0 for comparison if C is in free_symbols
                            # Or, more robustly, check if the derivative of the difference is 0
                            diff_expr = sp.simplify(user_expr_sympy - correct_expr_sympy)
                            is_constant_diff = False
                            if hasattr(diff_expr, 'is_constant'): # Sympy >= 1.6
                                if diff_expr.is_constant():
                                    is_constant_diff = True
                                else: # If not directly constant, check if derivative is zero
                                    is_constant_diff = (sp.diff(diff_expr, x) == 0)
                            else: # Older sympy fallback
                                is_constant_diff = (sp.diff(diff_expr, x) == 0)


                            is_correct_answer = is_constant_diff

                            st.write("### Answer Check:")
                            st.write(f"Your answer (parsed): ${sp.latex(user_expr_sympy)}$")
                            st.write(f"Correct antiderivative: ${correct_latex_display}$")

                            if is_correct_answer:
                                st.success("✅ Correct! Your answer is equivalent to the correct solution (up to a constant).")
                                state.correct += 1
                                state.points += state.level
                                st.write(f"Awesome! +{state.level} points!")

                                try:
                                    current_level_before_update = state.level
                                    new_potential_level = 1 # Default
                                    if LEVEL_UP:
                                        qualifying_lvls = [lvl for pts_thresh, lvl in LEVEL_UP.items() if state.points >= pts_thresh]
                                        if qualifying_lvls:
                                            new_potential_level = max(qualifying_lvls)

                                    if new_potential_level > current_level_before_update:
                                        st.balloons()
                                        st.success(f"🎉 Level Up! You're now at Level {new_potential_level}")
                                    state.level = new_potential_level
                                except Exception as level_err:
                                    st.error(f"Error calculating level: {level_err}")
                                    state.level = 1 # Fallback

                            else:
                                st.warning("❌ Incorrect. Your answer differs from the expected solution by more than just a constant.")
                                st.info(f"The correct antiderivative is: ${correct_latex_display}$")

                            with st.expander("Debug Details (Advanced)", collapsed=True):
                                st.write(f"Your raw input: `{answer_input}`")
                                st.write(f"Processed input for SymPy: `{processed_answer}`")
                                st.write(f"Parsed user expression (SymPy): `{user_expr_sympy}`")
                                st.write(f"Correct expression (SymPy): `{correct_expr_sympy}`")
                                st.write(f"Difference (User - Correct): `{diff_expr}`")
                                st.write(f"Is difference a constant? `{is_constant_diff}`")
                                st.write(f"Derivative of difference: `{sp.diff(diff_expr, x)}`")

                            # For the game flow, get a new problem after each attempt that is parsed successfully
                            state.prob = None
                            st.rerun()

                        except (SyntaxError, TypeError, sp.SympifyError) as parse_err:
                            st.error(f"⚠️ Could not parse your answer: `{parse_err}`")
                            st.info("Please check your SymPy syntax. Example: `x**2/2 + sin(x)`. Use `S(1)/3` for fractions like 1/3.")
                            # Do NOT get a new problem here, let user fix input.
                        except Exception as eval_err:
                            st.error(f"⚠️ Error evaluating your answer: {eval_err}")
                            st.info("There was an issue during the evaluation of your parsed expression.")


                    except Exception as solve_err:
                        st.error(f"⚠️ System error processing the problem's solution: {solve_err}")
                        st.info("An issue occurred with the current problem. A new problem will be loaded.")
                        state.prob = None
                        st.rerun()

            st.markdown("---")
            st.markdown(f"**Problems Solved:** {state.correct}/{state.answered}")

            with st.expander("Level Thresholds"):
                st.markdown("""
                | Points | Difficulty Level | Description |
                | ------ | --------------- | ----------- |
                | 0      | 🟢 Level 1      | Basic integrals (power rule, simple substitutions) |
                | 4      | 🟡 Level 2      | Intermediate (parts, trig identities, partial fractions) |
                | 8      | 🔴 Level 3      | Advanced (combined techniques, complex substitutions) |
                """)

            if st.button("🔄 Reset Game & Save Score"):
                if state.points > 0:
                    save_to_leaderboard()
                reset_game()

# ----------------- Leaderboard Tab -----------------
with tabs[1]:
    st.header("🏆 Leaderboard")
    if state.leaderboard.empty:
        st.info("No scores yet. Be the first to make it to the leaderboard!")
    else:
        st.dataframe(state.leaderboard, use_container_width=True, hide_index=True)
        if st.button("Clear Leaderboard", key="clear_leaderboard_button"):
            state.leaderboard = pd.DataFrame(columns=["Player", "Score", "Level", "Correct", "Total", "Accuracy", "Date"])
            st.success("Leaderboard cleared!")
            st.rerun()