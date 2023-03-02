# Severance package optimisation

In this assignment, the task is to develop a restructuring strategy for SFB. The assignment is based on the case description Integration Planning at SFB (Parts A, B, and C). The following tasks were completed:

### 1. Predicting the probability that an employee would take the severance offer

- Using Python, develop a model to predict the probability of employees at the Lyon facility accepting an RCC if it is offered to them.
- Output the predicted probability for each employee (e.g., as a list).
- What are the most important factors in determining whether an employee will accept an RCC?

### 2. Optimise the severance cost by offering the severance package to the right employee groups

- Formulate the problem outlined in Part C of the case as an optimization problem.
- Determine employee categories which may or may not be opened up to RCCs.
- Create group assignments based on the above and modify the optimization problem accordingly.
- Justify the choice of employee categories, especially in light of possible discrimination issues.
- Define the severance cost for each employee, based on the case description.
- Solve the optimization problem using Python. Note which groups are opened up to RCC, as well as the achieved severance cost.

### 3. Re-optimise under added constraints

- Add the following two constraints, and solve the optimisation problem again
  - (i) The average yearly amount gained from salary cuts must be greater than the savings expected by management and found in part A
  - (ii) In each of the three departments, at least 80% of the employees have to stay on.
- Ensure that, given the optimal solution, whether an employee is offered an RCC is not implicitly linked to sensitive variables such as gender. Make sure you describe your verification process.

### 4. Discussion

- Discuss possible pros and cons of the optimization approach. Discuss the assumptions and how likely are they to be fulfilled?
