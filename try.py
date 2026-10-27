import pandas as pd
import matplotlib.pyplot as plt
# Create a list of data to
# be represented in x-axis
subjects = [ 'Math' , 'English' , 'History' ,
            'Chem' , 'Geo' , 'Physics' , 'Bio' , 'CS' ]

# Create a list of data to be 
# represented in y-axis
stress = [ 9 , 3 , 5 , 1 , 8 , 5 , 10 , 2 ]

# Create second list of data
# to be represented in y-axis
grades = [ 15 , 10 , 7 , 8 , 11 , 8 , 17 , 20 ]

# Create a dataframe using the three lists
df = pd.DataFrame(list(zip( stress , grades )),
                  index = subjects , 
                  columns = ['Stress' , 'Grades' ])
df.index.name= 'Subject'
print(df)

ax = plt.gca()
df.plot(y = 'Stress', ax = ax )
df.plot(y = 'Grades' , ax = ax )
df.plot('Grades', 'Stress')
df.plot(color='red', title='Subjects')
plt.show()  