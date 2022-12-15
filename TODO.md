- predict probability of employee leaving
- select employee categories with the highest leaving prob and satisfy other requirements such as headcount targets using optimisation model
- severance package is a way to offer dissatisfied employees to leave

TODO:
- Check collinearity
- Missing values
  - Impute with model / mean
  - Can we discard them? Run model and check significance
- Features
  - Info on the RCC: RCC amount / RCC amount relative to daily rate, hourly rate, monthly income, monthly rate
- Severance package: we based this on last's time data, which is base + 3 month. We built a model based on that data -> For the probabilities to be comparable, need to use 3 months on top instead of 2. We could simulate what would happen if 2 were offered? should expect lower probability of leave. Maybe we can scale the leave prob with the severance package and see what happens? Philips: This is a pretty strong linearity assumption
  - --> analyse the effect of severance package using the model.
- Model interpretation: Does the model agree with the factors determining employee performance and intention to stay? Those factors are performance rating, YOE, job satisfaction
- According to Nikhita: The RCC serves as a way to nudge dissastisfied employees to leave. -> Analyse the final recommendation, how does employee satisfaction look like in the leaving vs remaining cohort?
- Test the initial assumptions: Longer ppl worked with the firm the higher the pay -> more likely to leave. But stronger attachment = less likely to leave.

Linear prog
- Constraints:
  - minimum number employees to fire: 40.
  - leaving ppl are evenly spread across job roles (with 2 -> 5%  tolerance)
  - The average yearly amount gained from salary cuts must be greater than the savings expected by management and found in part A (3 million euros);
  - In each of the three departments, at least 80% of the employees have to stay on.

- Other requirements
  - Do not target ppl based on some protected attributes (e.g., Gender)
  - The categories must not be too fine-grained that it targets 1 to 2 people

- Objective: minimise the severance package cost
- travel frequency:
  - if ppl in these department already traveled frequently -> can we not fire them and they will quit themselves? what if they don't quit and demand subsidies for travelling?`

# Future work
- Protected attributes: Could have added them to the optimisation model as additional constraints and optimise directly
