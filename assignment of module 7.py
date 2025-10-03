#Assingment 1

# Assignment 1 - Working with NumPy
# Objective Understand the basics of NumPy, array creation, and manipulation.
# Instructions 
# Step 1 Install NumPy using
# pip install numpy
# Step 2 Create a new Python script numpy_assignment.ipynb. Import NumPy and follow these steps:  
# 1. Create a 1D NumPy array with integers from 1 to 20. Perform the following operations:  
#   a. Calculate the sum, mean, median, and standard deviation of the elements in the array.  
#   b. Find the indices of elements greater than 10 in the array.  
# 2. Create a 2D NumPy array of shape 4 X 4 with numbers ranging from 1 to 16.  
#   a. Print the array.  
#   b. Find the transpose of the array.  
#   c. Calculate the row-wise and column-wise sums of the array.  
# 3. Create two 3 X 3 arrays filled with random integers between 1 and 20.  
#   a. Perform element-wise addition, subtraction, and multiplication.  
#   b. Compute the dot product of the two arrays.  
# 4. Reshape a 1D array of size 12 into a 3 X 4 2D array and slice the first two rows and last two columns.  

import numpy as np

arr_1d = np.arange(1,21)
arr_sum = np.sum(arr_1d)
arr_mean = np.mean(arr_1d)
arr_median = np.median(arr_1d)
arr_std = np.std(arr_1d)

print("1D Array:", arr_1d)
print("sum of all elements in 1D array:", arr_sum)
print("mean of all elements in 1D array:", arr_mean)
print("median of all elements in 1D array:", arr_median)
print('standard deviation of all elements in 1D array:', arr_std)

indices_gt_10 = np.where(arr_1d > 10)[0]
print("Indices of elements greater than 10 in 1D array:", indices_gt_10)
# 2. Create a 2D NumPy array of shape 4 X 4 with numbers ranging from 1 to 16.  
#   a. Print the array.  
#   b. Find the transpose of the array.
print("creating a 2d array of shape 4*4 with numbers ranging from 1 to 16 in arr_1d, we name this new 2D array as arr_2d")  
arr_2d = np.arange(1,17).reshape(4,4)
print("2D Array:", arr_2d)
print("Transpose of 2D Array:", arr_2d.T)
print("Row wise sum of 2D array:", np.sum(arr_2d, axis=1))
print("column wise sum of 2D array:", np.sum(arr_2d, axis=0))
# 3. Create two 3 X 3 arrays filled with random integers between 1 and 20.  
#   a. Perform element-wise addition, subtraction, and multiplication.  
#   b. Compute the dot product of the two arrays.  
print("creating two 3*3 arrays with random integers between 1 and 20, by naming them arr_a and arr_b")
arr_a = np.random.randint(1, 21, size=(3, 3))
arr_b = np.random.randint(1, 21, size=(3, 3))

print("Array A:", arr_a)
print("Array B:", arr_b)

print("Element wise Addition of Array A and Array B:", arr_a + arr_b)
print("Element wise Subtraction of Array A and Array B:", arr_a - arr_b)
print("Element wise Multiplication of Array A and Array B:", arr_a * arr_b)
print("Dot product of Array A and Array B:", np.dot(arr_a, arr_b))
print("Cross product of Array A and Array B:", np.cross(arr_a, arr_b))
# 4. Reshape a 1D array of size 12 into a 3 X 4 2D array and slice the first two rows and last two columns.  
print("Creating a reshaped 1D array of size 12 into a 3*4 2D array and slice the first two rows and last two columns")
arr_reshape = np.arange(1,13).reshape(3,4)
print("Reshaped 1D Array (3*4):", arr_reshape)
print("Sliced array (first two rows and last two columns):", arr_reshape[:2, -2:])


# Assignment 2 - Working with Pandas
# Objective Learn to create and manipulate dataframes for data analysis.  
# Instructions
# Step 1 Install Pandas using:
# pip install pandas
# Step 2 Create a new Python script pandas_assignment.ipynb. Import Pandas and follow these steps:  
# 1. Create a DataFrame with the following data:  
#   data = {
#       'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
#       'Age': [24, 27, 22, 32, 29],
#       'Department': ['HR', 'Finance', 'IT', 'Marketing', 'HR'],
#       'Salary': [45000, 54000, 50000, 62000, 47000]
#   }
#   a. Print the first five rows of the DataFrame.  
#   b. Get the summary statistics of the 'Age' and 'Salary' columns.  
#   c. Calculate the average salary of employees in the 'HR' department.  
# 2. Add a new column, 'Bonus', which is 10% of the salary.  
# 3. Filter the DataFrame to show employees aged between 25 and 30.  
# 4. Group the data by 'Department' and calculate the average salary for each department.  
# 5. Sort the DataFrame by 'Salary' in ascending order and save the result to a new CSV file.  

import pandas as pd
print("creating a dataframe with given data using pandas")
data = {
       'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
       'Age': [24, 27, 22, 32, 29],
       'Department': ['HR', 'Finance', 'IT', 'Marketing', 'HR'],
       'Salary': [45000, 54000, 50000, 62000, 47000]
       }
df = pd.DataFrame(data)
print("Dataframe:",df.head())
print("Creating summary statistics of Age and salary columns")
print("Summary statistics of Age and Salary columns:", df[['Age', 'Salary']].describe())
print("Calculating average salary of employees in HR department")
hr_avg_salary = df[df['Department'] == 'HR']['Salary'].mean()
print("Average salary of employees in HR department:", hr_avg_salary)
print("Creating a new column 'Bonus' which is 10% of the salary")
df['Bonus'] = df['Salary']*0.10
print("Dataframe with Bonus column:", df)
print("Filtering the dataframe to show employees aged between 25 and 30")
filtered_df = df[df['Age'].between(25, 30)]
print("Filtered Dataframe (Age between 25 and 30):", filtered_df)
print("Grouping the data by Department and calculating average salary for each department")
avg_salary_by_dept = df.groupby('Department')['Salary'].mean()
print("Average salary by Department:", avg_salary_by_dept)
print("Sorting the dataframe by Salary in ascending order and saving it to a new CSV file")
sorted_df = df.sort_values(by='Salary')
print("Sorted dataframe by salary in ascending order:", sorted_df) 
sorted_df.to_csv('sorted_employees.csv', index=False)
print("Sorted dataframe saved to 'sorted_employees.csv'")
# Assignment 3 - Working with Matplotlib
# Objective Practice data visualization techniques for better data representation.  
# Instructions 
# Step 1 Install Matplotlib using:
# pip install matplotlib
# Step 2 Create a new Python script matplotlib_assignment.ipynb. Import Matplotlib and follow these steps:  
# 1. Create a simple line plot for the following data:
#   x = [1, 2, 3, 4, 5]
#   y = [10, 15, 25, 30, 50]
#   a. Plot the data.  
#   b. Customize the plot by adding a title, axis labels, and a grid.  
# 2. Create a bar graph to represent the marks scored by students in a subject:  
#   students = ['John', 'Jane', 'Alice', 'Bob']
#   marks = [75, 85, 60, 90]
#   a. Plot the data as a bar graph.  
#   b. Customize the colors and add a title.  
# 3. Create a pie chart to represent the percentage distribution of a company’s revenue from different regions:  
#   regions = ['North America', 'Europe', 'Asia', 'Others']
#   revenue = [45, 25, 20, 10]
#   a. Create a pie chart with the region names as labels.  
#   b. Highlight the region with the highest revenue.  
# 4. Generate a histogram to show the frequency distribution of randomly generated integers between 1 and 100 (sample size = 1000).  
import matplotlib.pyplot as plt
import numpy as np 
x = [1,2,3,4,5]
y = [10,15,25,30,50]
plt.figure(figsize=(8,6))
plt.plot(x,y, marker='o', color='blue')
plt.title('simple line plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()
# 2. Create a bar graph to represent the marks scored by students in a subject:  
#   students = ['John', 'Jane', 'Alice', 'Bob']
#   marks = [75, 85, 60, 90]
#   a. Plot the data as a bar graph.  
#   b. Customize the colors and add a title.  
print("Creating a bar graph to represent the marks scored by students in subject Science")
students= ['John', 'Jane', 'Alice', 'BOb']
marks = [75, 85, 60, 90]
plt.figure(figsize=(8,6))
plt.bar(students, marks, color=['red', 'blue', 'green', 'orange'])
plt.title('marks scored by students in subject Science')
plt.xlabel('students')
plt.ylabel('marks')
plt.show()
# 3. Create a pie chart to represent the percentage distribution of a company’s revenue from different regions:  
#   regions = ['North America', 'Europe', 'Asia', 'Others']
#   revenue = [45, 25, 20, 10]
#   a. Create a pie chart with the region names as labels.  
#   b. Highlight the region with the highest revenue.  
print("Creating a pie chart to represent the precentage distribution of company's revenue from different regions")
regions = ['North America', 'Eorope', 'Asia', 'Others']
revenue = [45, 25, 20, 10]

explode = [0.1 if r== max(revenue) else 0 for r in revenue]
plt.figure(figsize=(8,6))
plt.pie(revenue, labels=regions, autopct='%1.1f%%', explode=explode, startangle= 140)
plt.title("Company's revenue from different regions")
plt.axis('equal')
plt.show()
# 4. Generate a histogram to show the frequency distribution of randomly generated integers between 1 and 100 (sample size = 1000).
print("Creating a histogram to show the frequency distribution of randomly generated integers between 1 and 100 (sample size = 1000)")
random_data = np.random.randint(1, 101, 1000)

plt.figure(figsize=(8, 5))
plt.hist(random_data, bins=20, color='purple', edgecolor='black')
plt.title("Frequency Distribution of Random Numbers")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
