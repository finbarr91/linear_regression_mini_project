import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn

import seaborn as sns

# special matplotlib argument for improved plots
from matplotlib import rcParams
sns.set_style("whitegrid")
sns.set_context("poster")

from sklearn.datasets import load_boston
import pandas as pd

boston = load_boston()

print(boston.keys())
print(boston.data.shape)
# Print column names
print(boston.feature_names)

# Print description of Boston housing data set
print(boston.DESCR)
# Now let's explore the data set itself.
bos = pd.DataFrame(boston.data)
print(bos.head())

bos.columns = boston.feature_names
print(bos.head())

# Now we have a pandas DataFrame called bos containing all the data we want to use to
# predict Boston Housing prices. Let's create a variable called PRICE which will contain the prices.
# This information is contained in the target data.
print(boston.target.shape)
bos['PRICE'] = boston.target
print(bos.head())

# EDA and Summary Statistics
# Let's explore this data set. First we use describe() to get basic summary statistics for each of the columns.
print(bos.describe())


# Scatterplots
# Let's look at some scatter plots for three variables: 'CRIM' (per capita crime rate), 'RM' (number of rooms) and 'PTRATIO' (pupil-to-teacher ratio in schools).

plt.scatter(bos.CRIM, bos.PRICE)
plt.xlabel("Per capita crime rate by town (CRIM)")
plt.ylabel("Housing Price")
plt.title("Relationship between CRIM and Price")
plt.show()

"""
Describe relationship
Answer = 1)CRIM is per capita crime rate by town. 
When crime rate is high, the housing price usually is on the low side.And the viceversa also is also true i.e when the crime rate is low, the price seems to be on the higer side. 
As crime rate increases from 0 thru 80, the housing price reduces. Thus CRIM has a negative linear relationship. 
Even when the CRIM is at 0, Some of the housing prices are on lower side. So I beleive there are other attributes impacting the housing price in these instances.
"""

# your turn: scatter plot between *RM* and *PRICE*
plt.scatter(bos.RM, bos.PRICE)
plt.xlabel("Average number of rooms per dwelling (RM)")
plt.ylabel("Housing Price")
plt.title("Relationship between Rm and Price")
plt.show()

# your turn: scatter plot between *PTRATIO* and *PRICE*
plt.scatter(bos.PTRATIO, bos.PRICE)
plt.xlabel("Pupil-to-Teacher Ratio (PTRATIO)")
plt.ylabel("Housing Price")
plt.title("Relationship between PTRATIO and Price")
plt.show()

# your turn: create some other scatter plots
plt.scatter(bos.LSTAT, bos.PRICE)
plt.xlabel("% lower status of population (LSTAT)")
plt.ylabel("Housing Price")
plt.title("Relationship between LSTAT and Price")
plt.show()


# Scatterplots using Seaborn
# Seaborn is a cool Python plotting library built on top of matplotlib. It provides convenient syntax and shortcuts for many common types of plots, along with better-looking defaults.
# We can also use seaborn regplot for the scatterplot above. This provides automatic linear regression fits (useful for data exploration later on). Here's one example below.

sns.regplot(y="PRICE", x="RM", data=bos, fit_reg = True)
plt.show()


# Histograms
plt.hist(np.log(bos.CRIM))
plt.title("CRIM")
plt.xlabel("Crime rate per capita")
plt.ylabel("Frequencey")
plt.show()

#your turn
plt.hist(bos.CRIM)
plt.title("CRIM")
plt.xlabel("Crime rate per capita")
plt.ylabel("Frequencey")
plt.show()

# In the plot without log transformation, the date looks highly right skewed.
# It is difficult to understand the data distribution as the crime rate after 20 is barely visible.
# But in the prior plot with log transformation, the plot looks more readable and allows us to understand the distribution better. The log transformation allows to show highly skewed data, as less skewed.



plt.hist(bos.RM)
plt.hist(bos.PTRATIO)
plt.title("RM vs PTRATIO ")
plt.xlabel("Crime rate and PTRATIO freq")
plt.ylabel("Frequencey")
plt.show()

# Import regression modules
import statsmodels.api as sm
from statsmodels.formula.api import ols

# statsmodels works nicely with pandas dataframes
# The thing inside the "quotes" is called a formula, a bit on that below
m = ols('PRICE ~ RM',bos).fit()
print(m.summary())

# your turn
predicted_price = m.fittedvalues
orig_price = bos.PRICE
sns.regplot(x=predicted_price,y=orig_price)

#
# Fitting Linear Regression using sklearn
from sklearn.linear_model import LinearRegression
X = bos.drop('PRICE', axis = 1)

# This creates a LinearRegression object
lm = LinearRegression()

# Fit a linear model

# The lm.fit() function estimates the coefficients the linear regression using least squares.

# Use all 13 predictors to fit linear regression model
lm.fit(X, bos.PRICE)

"""Part 3 Checkup Exercise Set II

Exercise: How would you change the model to not fit an intercept term? Would you recommend not having an intercept? Why or why not? For more information on why to include or exclude an intercept, look [here](https://stats.idre.ucla.edu/other/mult-pkg/faq/general/faq-what-is-regression-through-the-origin/).

Exercise: One of the assumptions of the linear model is that the residuals must be i.i.d. (independently and identically distributed).To satisfy this, is it enough that the residuals are normally distributed? Explain your answer.

Exercise: True or false. To use linear regression,

must be normally distributed. Explain your answer.

Answer:We can set the parameter fit_intercept = False, to ensure no intercept is used during calculation. It is possible to have a model without a intercept. But it means that the slope is forced to go through the origin and R2 value becomes higher artificially, not because the model is better.

Answer :Yes when residuals are i.i.d it means that they share the same propability distribution. Same probabilty distribution means they resemble normal distribution.

Answer:False, Y need not be normally distributed. The assumption is true ony for residuals.
"""

print('Estimated intercept coefficient: {}'.format(lm.intercept_))
print('Number of coefficients: {}'.format(len(lm.coef_)))
# The coefficients
print(pd.DataFrame({'features': X.columns, 'estimatedCoefficients': lm.coef_})[['features', 'estimatedCoefficients']])

# first five predicted prices
print(lm.predict(X)[0:5])
"""

Part 3 Checkup Exercise Set III

Exercise: Histogram: Plot a histogram of all the predicted prices. Write a story about what you see. Describe the shape, center and spread of the distribution. Are there any outliers? What might be the reason for them? Should we do anything special with them?

Answer:The histogram of predicted price almost looks like is normal symmetric, single peak distribution. But there is a mild skewness towards left. The center of the distribution by looking at the histogram seems to be between 20 and 25. We can confirm that by finding the mean value which is 22.53. Approximate range derived from histogram is 45 - (-5) = 50 , where 45 is approximate max value and -5 is approximate min value. There is one point less that is negative which can be considered outlier.

Exercise: Scatterplot: Let's plot the true prices compared to the predicted prices to see they disagree (we did this with `statsmodels` before).

Exercise: We have looked at fitting a linear model in both `statsmodels` and `scikit-learn`. What are the advantages and disadvantages of each based on your exploration? Based on the information provided by both packages, what advantage does `statsmodels` provide?

Answer:Statsmodels: We get nice summarized statistical information. For e.g. the p-value along with confidence intervals lets us know if a feature is significant predictor or not. This tool seems to be built for statisticians. But documentation wise it is difficult to understand. Sklearn: This seems more beginner friendly with good documentation. But we have only basic information. If we need to do much more detailed statistical analysis to understand the model better, then Statsmodel is better.

"""
# your turn
predicted_prices = lm.predict(X)
predicted_prices_df = pd.DataFrame(predicted_prices)
predicted_prices_df.describe()

predicted_prices_df.hist()
plt.show()

print(np.sum((bos.PRICE - lm.predict(X)) ** 2))
print(np.sum((lm.predict(X) - np.mean(bos.PRICE))**2))

"""
Part 3 Checkup Exercise Set IV

Let's look at the relationship between `PTRATIO` and housing price.

Exercise: Try fitting a linear regression model using only the 'PTRATIO' (pupil-teacher ratio by town) and interpret the intercept and the coefficients.

Exercise: Calculate (or extract) the R2 value. What does it tell you?

Exercise: Compute the F-statistic. What does it tell you?

Exercise: Take a close look at the F-statistic and the t-statistic for the regression coefficient. What relationship do you notice? Note that this relationship only applies in *simple* linear regression models.

Answer:1)
Coefficient: A one unit increase in PTRATIO means a decrease in housing by approximately 2157. The confidence interval for change is between 2477 and 1837. Intercept: When PTRATIO coefficient is 0, then expected price is 62. p-value : It is close to 0, and hence PTRATIO is considered statistically significant. 2)R-squared is a statistical measure of how close the data are to the fitted regression line. R2 is 0.258. This means only 25.8 % of variability in price is explained by the model 3)F-statistic is 175.1. F-test compares the model we built to the intercept only model and decides whether adding coeffecient improves our model. In our case it means that PTRATIO is significantly improving the model. 4)T-statistic = - 13.233. F-statistic = 175.1. 
Looks like F-statistic is t-statistic-Squared.
"""
# your turn
m1 = ols('PRICE ~ PTRATIO',bos).fit()
print(m1.summary())

"""
Part 3 Checkup Exercise Set V

Fit a linear regression model using three independent variables

    'CRIM' (per capita crime rate by town)
    'RM' (average number of rooms per dwelling)
    'PTRATIO' (pupil-teacher ratio by town)

Exercise: Compute or extract the F-statistic. What does it tell you about the model?

Exercise: Compute or extract the R2 statistic. What does it tell you about the model?

Exercise: Which variables in the model are significant in predicting house price? Write a story that interprets the coefficients.
"""
# your turn
m2 = ols('PRICE ~ CRIM + RM + PTRATIO',bos).fit()
print(m2.summary())

# Part 4 Checkup Exercises
#
# Exercise: Find another variable (or two) to add to the model we built in Part 3. Compute the
#
# F-test comparing the two models as well as the AIC. Which model is better?

m3 = ols('PRICE ~ CRIM + RM + PTRATIO+LSTAT+AGE',bos).fit()
print(m3.summary())
"""
Answer:Comparing the AIC between two models, AIC of M3 has come down, indicating this model is better fit for the data. Also p-values for all of the variables are close to zero. F-statistic also is higher which means that overall this model is better in predicting the price.

M2 model (CRIM + RM + PTRATIO) F-statistic = 244.2 AIC = 3233

M3 model (CRIM + RM + PTRATIO + AGE+ B) F-statistic = 215.9 AIC = 3110
"""

"""

Part 5 Checkup Exercises

Take the reduced model from Part 3 to answer the following exercises. Take a look at [this blog post](http://mpastell.com/2013/04/19/python_regression/) for more information on using statsmodels to construct these plots.

Exercise: Construct a fitted values versus residuals plot. What does the plot tell you? Are there any violations of the model assumptions?

Exercise: Construct a quantile plot of the residuals. What does the plot tell you?

Exercise: What are some advantages and disadvantages of the fitted vs. residual and quantile plot compared to each other?

Exercise: Identify any outliers (if any) in your model and write a story describing what these outliers might represent.

Exercise: Construct a leverage plot and identify high leverage points in the model. Write a story explaining possible reasons for the high leverage points.

Exercise: Remove the outliers and high leverage points from your model and run the regression again. How do the results change?

"""

# Your turn.
residuals = np.array((bos.PRICE - predicted_prices))
print(residuals)
print("Residual mean",residuals.mean())
plt.scatter(predicted_prices,residuals)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title('Residuals vs Fitted')
plt.show()

# Answer
# 1): Residual plot are useful graphical tool for identifying non-linearity. Ideally residual plot will show no discrenible pattern. A presense of pattern indicate some problem with some aspect of the linear model. Observing the , the residual exhibit a slight u shape which provides indication of some level of non-linearity in the data.
# This is in violation of the model assumption Normal Distribution of errors

model_norm_residuals = m2.get_influence().resid_studentized_internal

from statsmodels.graphics.gofplots import ProbPlot
QQ = ProbPlot(model_norm_residuals)

Quantile_plots = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)
Quantile_plots.axes[0].set_title('Normal Q-Q')
Quantile_plots.axes[0].set_xlabel('Theoretical Quantiles')
Quantile_plots.axes[0].set_ylabel('Standardized Residuals');



# Answer 2):A Q-Q plot is a scatterplot created by plotting two sets of quantiles against one another. If both sets of quantiles came from the same distribution, we should see the points forming a line that’s roughly straight.Looking at the graph above, there are several points that fall far away from the red line. This is indicative of the errors not being normally distributed. This means that our data has more extreme values or outliers that would be expected if they truely came from a normal distribution.
#
# Answer 3):Both the plots help to identify the outliers and help to understand whether residuals are from a normal distribution. Apart from above, q-q plot also helps to answer whether two datasets have common location and scale, have similar distributional shape and have similar tail behavior.
#
# Answer 4):Based on the residuals plot, ther are 7 outliers.

from statsmodels.graphics.regressionplots import *
import statsmodels.api as sm
fig,ax  = plt.subplots(figsize=(8,8))
fig = sm.graphics.plot_leverage_resid2(m2, ax=ax)
plt.show()

fig, ax  = plt.subplots(figsize=(12,10))
fig = sm.graphics.influence_plot(m2, ax=ax)
plt.show()

print(bos[380:381])

print(lm.predict(X[380:381]))

X_Upd = bos.drop([380, 418,405, 364, 365, 367, 368, 369, 370, 371, 372],axis=0)
print(X_Upd[364:381])

m4 = ols('PRICE ~ CRIM + RM + PTRATIO',X_Upd).fit()
print(m4.summary())

# Answer
# 5):After removing the outliers and high leverage points, re-runing the model has significant increase in R-squared








