library(cvTools)
fnm  <- "/Users/vema/Documents/pyscr/datasets/bank/bank.csv"
X <- read.csv(fnm, header=T, sep=";")
N = nrow(X)
k = 10
kf = cvFolds(N, K=k, type="consecutive") 

aucvec  <-  rep(0, k)
for (i in 1:k) {
  trainData <- X[kf$which != i,]
  testData <- X[kf$which == i,]
  print(c(nrow(trainData), nrow(testData)))
  glmobj  <-  glm(y~., data=trainData, family=binomial)
  pred <- predict(glmobj, testData, type="link" )
  fitpred = prediction(pred, testData$y)
  auc.tmp <- performance(fitpred,"auc"); 
  auc <- as.numeric(auc.tmp@y.values)
  aucvec[i] <- auc
}

