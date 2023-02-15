# Disease_prediction
ML project Repository

# Introduction:

Disease prediction is an important field of study, as early detection and accurate
prediction can aid in timely medical intervention and improve patient outcomes. In
recent years, machine learning techniques have been applied to disease prediction,
showing promising results. This report provides an overview of disease prediction using
machine learning

# Methodology -

Data Preprocessing - Data had 134 columns so we checked for the Null columns and
removed them. Also done the required division of data for the further processing.
LabelEncoder - Here we are using the LabelEncoder class from the sci-kit-learn library.
The LabelEncoder is used to convert categorical data (such as disease names) into
numerical data that can be used in statistical machine learning models.
In this example, the LabelEncoder is applied to the "prognosis" column of a DataFrame
called df. First, the fit method is used to "fit" the LabelEncoder to the "prognosis" column
of the DataFrame. This step involves learning the mapping between the unique values
in the "prognosis" column and their corresponding numerical codes.
Once the LabelEncoder is fitted to the "prognosis" column, the transform method is
used to apply the numerical codes to the values in the "prognosis" column.

sns.heatmap(): This is the seaborn function used to create a heatmap.
df.isnull(): This returns a DataFrame of the same shape as df but with boolean values
where there are missing values (NaN).
cmap='viridis': This sets the color map for the heatmap to 'viridis', which is a color map
that goes from yellow to green to blue, representing increasing values.
cbar=False: This removes the color bar from the heatmap, which is typically used to
show the color scale for the values in the heatmap.

mask[np.triu_indices_from(mask)] = True:
This sets the upper triangle of the mask array to True, so that only the lower triangle of
the correlation matrix is shown in the heatmap. This is done to avoid showing redundant
information and make the heatmap easier to read.
f, ax = plt.subplots(figsize=(15,15),dpi=200):
This creates a matplotlib figure with a size of 15x15 inches and a DPI of 200. The f
variable contains the figure object and the ax variable contains the axis object.
cmap = sns.diverging_palette(220,10,as_cmap=True):
This creates a custom colormap that goes from blue (220) to red (10) via white. The
as_cmap=True argument returns the colormap as a matplotlib colormap object.

data = {'Symptoms': [], 'Prognosis': [], 'length': []}:
This creates a Python dictionary with keys for the columns that will be in the DataFrame
(Symptoms, Prognosis, and length).
table = table.astype({"Symptoms": str, "Prognosis": object, 'length': int}):
This changes the data types for each of the columns in the table DataFrame.
i = 0: This initializes a counter variable called i.
prognosis = df[df[symp] == 1].prognosis.unique().tolist(): This extracts a list of
unique prognoses associated with the current symptom by filtering the original dataset
df to find all rows where the current symptom is present (df[symp] == 1) and then
extracting the unique values in the prognosis column of those rows.
table.sort_values(by='length', ascending=False).head(10):
This sorts the table DataFrame by the length column in descending order and then
returns the top 10 rows, which represent the symptoms that are most strongly
associated with a large number of prognoses.
from sklearn.tree import DecisionTreeClassifier:
This imports the DecisionTreeClassifier class from the scikit-learn library, which is used
to build decision tree models.
random_state=42: This creates a new instance of the DecisionTreeClassifier class and
assigns it to the variable dis. The random_state=42 argument sets the random seed for
the model to ensure that the results are reproducible.
coeff = np.mean(log.coef_,axis=0): This calculates the mean value of the coefficients for
each feature in the logistic regression model and assigns the resulting array to the
variable coeff.
threshold = np.quantile(np.abs(coeff), q=0.25): This calculates the 25th percentile of the
absolute values of the coefficients and assigns it to the variable threshold.


# Conclusion:

We observed 100% accuracy in all the models which is hinting toward the overfitting of
our model. Hence, we decided to analyze the data further in order to solve this problem.
We find the weights of all the features in our model to evaluate their importance.
We dropped the features with a weight of less than a certain threshold. After
successfully dropping, we were left with 99 features. Now we apply all the models to the
data with reduced features.
The key findings of the study highlighted the potential of machine learning algorithms in
disease prediction. By using medical and clinical data, as well as genetic and
environmental factors, these algorithms can accurately predict the occurrence and
severity of various diseases
