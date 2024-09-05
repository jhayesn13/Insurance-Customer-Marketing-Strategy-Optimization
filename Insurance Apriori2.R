#Importing data sets
df <- read.csv("D:/2023/School/St Johns/Fall 2023/Data Mining/Project Insurance/insuranceDataWClusters.csv")
View(df) #with classes
df1 <- read.csv("D:/2023/School/St Johns/Fall 2023/Data Mining/Project Insurance/insuranceDataWClustersDummy.csv")
View(df1) #dummy columns

#install relevant libraries
library(party)
library(caret)
library(e1071)
library(randomForest)
library(arules)
library(Cairo)

#Pre-processing
for (col in names(df)) {
  df[[col]] <- as.factor(df[[col]])
}
summary(df)

for (col in names(df1)) {
  df1[[col]] <- as.factor(df1[[col]])
}
summary(df1)



#Decision Trees
clusterTree <- ctree(Cluster ~ gender_female + gender_male + diabetic_No + diabetic_Yes + smoker_No +
                       smoker_Yes + children_0 + children_1 + children_2 + children_3 + children_4 +
                       children_5 + region_northeast + region_northwest + region_southeast + 
                       region_southwest + wtClass_Class.1.Obese + wtClass_Class.2.Obese +
                       wtClass_Class.3.Obese + wtClass_Healthy + wtClass_Overweight + 
                       wtClass_Underweight +
                       ageClass_Middle.Aged.Adult + ageClass_Older.Adult + ageClass_Young.Adult, data=df1)
print(clusterTree)
plot(clusterTree)

#Print Decision Tree
png("D:/2023/School/St Johns/Fall 2023/Data Mining/Project Insurance/clusterTree.png", 
    res=80, height=2000, width=6500) 
  plot(clusterTree) 
dev.off()

#Decision Tree
set.seed(1234)
sampleIndex <- sample(2, nrow(df1), replace = TRUE, prob=c(0.7, 0.3))
trainData <- df1[sampleIndex==1,]
testData <- df1[sampleIndex==2,]

treeSpecs <- Cluster ~ gender_female + gender_male + diabetic_No + diabetic_Yes + smoker_No +
  smoker_Yes + children_0 + children_1 + children_2 + children_3 + children_4 +
  children_5 + region_northeast + region_northwest + region_southeast + 
  region_southwest + wtClass_Class.1.Obese + wtClass_Class.2.Obese +
  wtClass_Class.3.Obese + wtClass_Healthy + wtClass_Overweight + 
  wtClass_Underweight +
  ageClass_Middle.Aged.Adult + ageClass_Older.Adult + ageClass_Young.Adult

#Tree predictions
predictTree <- ctree(treeSpecs, data=trainData)
predictions <- predict(predictTree)
confusionMatrix(predictions, trainData$Cluster)

#Naive Bayes
ncol(df1) #31 columns
classifier <- naiveBayes(df1[,2:31], df1[,1])
table(predict(classifier, df1[,-1]), df1[,1])

#Naive Bayes prediction
predictions1 <- predict(classifier, newdata = trainData)
confusionMatrix(predictions1, trainData$Cluster)

#Random Forest
rfModel <- train(
  Cluster ~ .,
  data = df1,
  method = "rf",
  trControl = trainControl(method = "cv", number = 10)
)

#Random Forest Predictions
predictions2 <- predict(rfModel, newdata = trainData)
confusionMatrix(predictions2, trainData$Cluster)
plot(rfModel)

#Apriori
df <- df[, names(df) != "bpClass"]
df <- df[, names(df) != "claimClass"]

rules <- apriori(df, parameter = list(supp = 0.5, conf = 0.9, target = "rules"))
inspect(head(rules, 15))

#High lift rules
rules1 <- apriori(df,
                  parameter = list(minlen=2, supp=0.1, conf=0.5),
                  appearance = list(rhs=c(
                    "Cluster=0",
                    "Cluster=1",
                    "Cluster=2",
                    "Cluster=3"),
                    default="lhs"),
                  control = list(verbose=F))
rules1 <- sort(rules1, by="lift", decreasing=TRUE)
inspect(head(rules1, 200))

rules2 <- apriori(df,
                  parameter = list(minlen=2, supp=0.05, conf=0.3),
                  appearance = list(rhs=c(
                    "Cluster=1",
                    "Cluster=3"),
                    default="lhs"),
                  control = list(verbose=F))
rules2 <- sort(rules2, by="lift", decreasing=TRUE)
inspect(head(rules2, 200))

#High Confidence Rules
rules1HC <- apriori(df, #High Confidence
                  parameter = list(minlen=2, supp=0.1, conf=0.8),
                  appearance = list(rhs=c(
                    "Cluster=0",
                    "Cluster=1",
                    "Cluster=2",
                    "Cluster=3"),
                    default="lhs"),
                  control = list(verbose=F))
rules1HC <- sort(rules1HC, by="confidence", decreasing=TRUE)
inspect(head(rules1HC, 200))


rules2HC <- apriori(df, #High Confidence
                  parameter = list(minlen=2, supp=0.05, conf=0.7),
                  appearance = list(rhs=c(
                    "Cluster=1",
                    "Cluster=3"),
                    default="lhs"),
                  control = list(verbose=F))
rules2HC <- sort(rules2HC, by="confidence", decreasing=TRUE)
inspect(head(rules2HC, 20))
