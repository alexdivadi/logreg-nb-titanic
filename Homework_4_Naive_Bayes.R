#title: "Homework 4: Naive Bayes"
#subtitle: "4375 Machine Learning with Dr. Mazidi"
#author: "David Allen"
#date: "Feb 24, 2022"

# read in CSV file
df <- read.csv('titanic_project.csv', header=TRUE)
head(df)

df$sex <- factor(df$sex)
df$pclass <- factor(df$pclass)
df$survived <- factor(df$survived)

# separate train and test
i <- 1:900
train <- df[i,]
test <- df[-i,]

# machine learning part here (train model)
library(e1071)
start <- proc.time()
nb1 <- naiveBayes(survived~pclass+sex+age, data=train)
end <- proc.time()

# Summary of model
nb1
end - start

# get predictions
pred <- predict(nb1, newdata=test, type="class")
table(pred, test$survived)

# get metrics
acc <- mean(pred==test$survived)
sensitivity <- sum(pred==1 & test$survived==1)/sum(pred==1)
specificity <- sum(pred==0 & test$survived==0)/sum(pred==0)

# print results
print(paste("Accuracy: ", acc))
print(paste("Sensitivity: ", sensitivity))
print(paste("Specificity: ", specificity))