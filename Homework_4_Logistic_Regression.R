#title: "Homework 4: Logistic Regression"
#subtitle: "4375 Machine Learning with Dr. Mazidi"
#author: "David Allen"
#date: "Feb 24, 2022"

# read in CSV file
df <- read.csv('titanic_project.csv', header=TRUE)
head(df)

# separate train and test
i <- 1:900
train <- df[i,]
test <- df[-i,]

# machine learning part here (train model)
start <- proc.time()
glm1 <- glm(survived~pclass, data=train, family="binomial")
end <- proc.time()

# Summary of model
summary(glm1)
end - start

# get predictions
probs <- predict(glm1, newdata=test, type="response")
pred <- ifelse(probs>0.5, 1, 0)
table(pred, test$survived)

# get metrics
acc <- mean(pred==test$survived)
sensitivity <- sum(pred==1 & test$survived==1)/sum(pred==1)
specificity <- sum(pred==0 & test$survived==0)/sum(pred==0)

# print results
print(paste("Accuracy: ", acc))
print(paste("Sensitivity: ", sensitivity))
print(paste("Specificity: ", specificity))