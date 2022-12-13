In this assignment, you are asked to develop a restructuring strategy for SFB. The assignment is based on the case description Integration Planning at SFB (Parts A, B, and C), found in the Tutorial 4 materials, together with the relevant data and the file Starting_model.xlsx. You are required to complete the following tasks:

- [x] Using Python, complete the prediction task outlined in Part B of the case. In particular, develop a model that allows you to predict the probability of employees at the Lyon facility accepting an RCC if it is offered to them.
- [] Make sure that your code prints out the predicted probability for each employee (e.g., as a list).
- [] What are the most important factors in determining whether an employee will accept an RCC?
- [] Formulate the problem outlined in Part C of the case as an optimization problem. Be sure to clarify what are the key elements of the problem as learned in class.
- [] Open Starting_model.xlsx. Each employee is given with a probability of accepting an RCC (please use this probability, not the one you predicted). Determine employee categories which may or may not be opened up to RCCs. In the xlsx file, employees have been assigned at random to placeholder categories
- [] make sure to overwrite the group assignments based on your choices and modify the optimization problem accordingly. Moreover, make sure to justify your choice of employee categories, especially in light of possible discrimination issues.
- [x] In your Solver model, define the severance cost for each employee (the current values are randomly generated), based on the case description.
- [] Solve the optimization problem using Excel Solver. Be sure to note which groups are opened up to RCC, as well as the cost you achieve.
- [] The optimization problem is incomplete. Add the following two constraints: (i) The average yearly amount gained from salary cuts must be greater than the savings expected by management and found in part A; (ii) In each of the three departments, at least 80% of the employees have to stay on. With the additional constraints, solve the problem again. How do the decision variables and the objective change?
- [] Ensure that, given the optimal solution, whether an employee is offered an RCC is not implicitly linked to sensitive variables such as gender. Make sure you describe your verification process.
- [] Discuss possible pros and cons of the optimization approach. What are the assumptions you are making implicitly, and how likely are they to be fulfilled?

By December 16 (4pm), you are supposed to submit the following package:
A notebook, containing all the code required to reproduce your Python analysis.
An xlsx-file, containing the optimization problem and solution you set up (based on Starting_model.xlsx).
Your textual answers, including justifications. You can either submit these in a separate pdf-file, or as part of your notebook. In either case, make clear which question you are answering
