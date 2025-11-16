import pandas as pd

#data frame with training data
train_df = pd.read_csv('train.csv')

#print(train_df)
#print(train_df.head())
#print(train_df.info())
#print(train_df['Survived'].value_counts())

"""first_class = train_df[train_df['Pclass']==1]
print(len(first_class))
print(train_df['Pclass'].value_counts())

female = train_df[train_df['Sex']=='female']
print(female)
print(female['Survived'].value_counts())

female_by_class = survivor_female['Pclass'].value_counts()
print(female_by_class)
"""

#Hypothesis 1: (Survivorship by sex)
print("Survivorship by sex")
print("Man: ")
male_survived = train_df[train_df['Sex']=='male']['Survived']
male_survived_percent = male_survived.mean()
print(f"{male_survived_percent:.2%}")

print("Women: ")
female_survived = train_df[train_df['Sex']=='female']['Survived']
female_survived_percent = female_survived.mean()
print(f"{female_survived_percent:.2%}")

#Hypothesis 2: (Survivorship by Pclass)
print("\nSurvivorship by Pclass")
print("First class: ")
first_class = train_df[train_df['Pclass']==1]['Survived']
first_class_percent = first_class.mean()
print(f"{first_class_percent:.2%}")

print("Second class: ")
second_class = train_df[train_df['Pclass']==2]['Survived']
second_class_percent = second_class.mean()
print(f"{second_class_percent:.2%}")

print("Third class: ")
third_class = train_df[train_df['Pclass']==3]['Survived']
third_class_percent = third_class.mean()
print(f"{third_class_percent:.2%}")

#Hypothesis 3: (Survivorship by age)
pass_without_age = train_df['Age'].isnull().sum()
all_pass_num = len(train_df)
print(f"\nPassengers with missing age data: {pass_without_age} out of {all_pass_num} ({(pass_without_age / all_pass_num):.1%})  ")

pass_with_age = train_df[train_df["Age"].notna()]

#0-12
children_percent = pass_with_age[pass_with_age['Age']<=12]['Survived'].mean()
teen_percent = pass_with_age[(pass_with_age['Age']>12) & (pass_with_age['Age']<18)]['Survived'].mean()
adults_percent = pass_with_age[(pass_with_age['Age']>18) & (pass_with_age['Age']<60)]['Survived'].mean()
elderly_percent = pass_with_age[pass_with_age['Age']>60]['Survived'].mean()
print("Survivorship by age")
print(f"Children (0-12): {children_percent:.1%}\n"
      f"Teenagers (13-17): {teen_percent:.1%}\n"
      f"Adults (18-59): {adults_percent:.1%}\n"
      f"Elderly (60+): {elderly_percent:.1%}\n")




