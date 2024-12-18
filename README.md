
# Sports Management Data Analysis

This project involves the analysis and preprocessing of a sports management dataset. The goal is to explore the data, preprocess it, and apply various machine learning models to evaluate and visualize the results.

[Click here for our Slides from Presentation](https://docs.google.com/presentation/d/1nvzgiq0oMXlzQAZmxxQAJeG7NjMHKxXoshiLME0NI8g/edit?usp=sharing)

## Project Structure

```

├── __pycache__/
├── .ipynb_checkpoints/
├── archive/
├── plots/
├── pipeline.py
├── Preprocessing.ipynb
├── README.md
├── sports_management_dataset.csv
├── sports_management_encoded.csv
├── sports_sustainability.ipynb

```




## **Summary Analysis and Results**  

### **Objective**  
This project explores the use of AI to assess and predict sustainability metrics in sports events using the Sustainable Sports Event Management (SSEM) dataset. The primary goal was to build a predictive model for the Sustainability Score and derive insights into the dataset's patterns and usability.  

### **Dataset Analysis**  
The SSEM dataset contains 102,000 samples with all features represented as categorical data, either ordinal or nominal. The target variable is the "Sustainability Score," classified as Low, Moderate, or High.  

Challenges encountered:  
- **Class Imbalance:** A skewed distribution of target classes led to difficulties in accurate prediction.  
- **Random Data Patterns:** Despite preprocessing and modeling efforts, the dataset exhibited no discernible relationships or patterns.  

### **Methodology**  
1. **Preprocessing:**  
   - Dropped any nulls (no null values were found in the dataset).
   - Encoded categorical features into numerical formats.    
   - Assessed multicollinearity using Variance Inflation Factor (VIF).  
2. **Modeling:**  
   - Trained a Gradient Boosting model due to its effectiveness with categorical data.  
   - Addressed class imbalance using random undersampling. We decided to do undersampling because we realized our dataset is randomly generated, oversampling could amplify this artificiality, reducing the reliability and interpretability of our model's performance.
   - Performed hyperparameter tuning to optimize model performance.  

### **Results**  
- **Initial Model:** The model achieved a balanced accuracy score of 33.3%.  
- **After Undersampling:** The accuracy score dropped to 33.2%.  
- **After Hyperparameter Tuning:** Balanced accurracy score came to 33.4%. Undersampling was not included in the tuned model as it didn't show any improvement in accuracy.

### **Insights**  
The results suggest that the SSEM dataset is likely generated randomly, lacking meaningful relationships between features and target variables. This significantly impacts the model's ability to learn and predict effectively.  

### **Conclusion**  
While the project underscored the importance of robust data preprocessing and modeling techniques, it also highlighted the critical role of realistic, high-quality datasets in AI research. Future work should focus on developing datasets with meaningful patterns or collaborating with sports organizations to gather real-world data for analysis.  



## Notebooks

- **Preprocessing.ipynb**: Data preprocessing steps including encoding and scaling.
- **sports_sustainability.ipynb**: Analysis and evaluation of sports sustainability metrics.
- **LearningCurveSustainability.ipynb**: Allows us to see the learning curve of the model.
## Data

- **sports_management_dataset.csv**: The original dataset containing various features related to sports management.
- **sports_management_encoded.csv**: Fully encoded dataset used for model training and evaluation.

## Usage

1. **Preprocessing**: Use `Preprocessing.ipynb` to understand the dataset and preprocess the data.
2. **Modeling**: Use 'pipeine.py' and `sports_sustainability.ipynb` to run the Gradient Boosting Model

## Requirements

- Python 3.12.4
- Jupyter Notebook
- pandas
- scikit-learn
- imbalanced-learn
- matplotlib
- pydotplus

## Installation

1. Clone the repository:
   ```sh
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```sh
   cd <project-directory>
   ```
3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
```
