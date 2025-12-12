# Assignment 6 Part 2 - Writeup

---

## Question 1: Feature Importance

Based on your house price model, rank the four features from most important to least important. Explain how you determined this ranking.

**YOUR ANSWER:**
1. Most Important: Bedrooms
2. Bathrooms
3. Age
4. Least Important: SquareFeet

**Explanation:**
I choose this ranking due to the absolute values of the coefficients since that shows the difference. The model gave bedrooms the highst and squarefeet the lowest abs value coefficient so thats why bedrooms is highest and square feet is lowest.
---

## Question 2: Interpreting Coefficients

Choose TWO features from your model and explain what their coefficients mean in plain English. For example: "Each additional bedroom increases the price by $___"

**Feature 1:** SquareFeet

Coefficient: 121.11
Meaning: Each bedroom, bathroom, age constant, additional square foot adds $121.11 to the predicted price

**Feature 2:** Bedrooms

Coefficient: 6648.97
Meaning: Each square footage, bathroom, age constant, additional bedroom increases the predicted price by $6,648.97

---

## Question 3: Model Performance

What was your model's R² score? What does this tell you about how well your model predicts house prices? Is there room for improvement?

**YOUR ANSWER:**
R² score: 0.9936 
Meaning:  The meaning if this is that about 99.36% of variance in the data given, so good fit.
Improvement: The dataset is small but clean so the improvement would be to have more data and more relevant features like location, condition, lot size etc.

---

## Question 4: Adding Features

If you could add TWO more features to improve your house price predictions, what would they be and why?

**Feature 1:** Location 

**Why it would help:** Location is very important, big city homes sell for more than rural homes.

**Feature 2:** Lot size

**Why it would help:** The size of the lot also brings value, outside of just the house square footage.

---

## Question 5: Model Trust

Would you trust this model to predict the price of a house with 6 bedrooms, 4 bathrooms, 3000 sq ft, and 5 years old? Why or why not? (Hint: Think about the range of your training data)

**YOUR ANSWER:**
I wouldnt fully trust this due to the data being outside of our range. And since the further you go outside of the range its less trustworthy prediction so less reliable. This is why its a maybe for me.



