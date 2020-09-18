## 10-flod MICE method
# Stratified k-fold cross-valiation using package


cpr_3_nomice <- read.csv("cpr_data_new1.csv",header = TRUE, sep = ",")
#cpr_3 <- read.csv("prob_df_10fold.csv",header = TRUE, sep = ",")
#totcpr_3 <- cpr_3[1:91]
folds <- 10
# handle different classifiers 
totcpr_3$output <- as.factor(totcpr_3$output)
#levels(totcpr_3$output) <- make.names(levels(factor(totcpr_3$output)))
cvIndex <- createFolds(factor(totcpr_3$output), folds, returnTrain = T)
# native bayes
levels(totcpr_3$output) <- make.names(levels(factor(totcpr_3$output)))
# convert output(0,1) to X0,X1 for nb classifiers
nb_model <- train(output~., data=totcpr_3, metric = "ROC", method="nb",
                  #ceprePross = c("center","scale"),
                  #tuneGrid = grid,
                  trControl=trainControl(method="repeatedcv", index = cvIndex, 
                                         number = folds, repeats = 5,
                                         classProbs = TRUE,verboseIter = FALSE,
                                         savePredictions = TRUE,
                                         summaryFunction = twoClassSummary) )

# logistic regression
glm_model <- train(output~., data=totcpr_3, method="glm", metric="ROC", 
                   #tuneGrid = expand.grid(alpha = 0:1, lambda = 0:10/10),
                   trControl=trainControl(method="repeatedcv", number=folds, 
                                          index = cvIndex, repeats = 5,
                                          classProbs = TRUE,verboseIter = FALSE,
                                          savePredictions = TRUE,
                                          summaryFunction = twoClassSummary) )
# random forest
rf_model <- train(output ~ ., data = totcpr_3, method = "ranger", metric="ROC",
                  #tuneGrid = rfGrid,
                  trControl=trainControl(method="repeatedcv", number=folds, 
                                         index = cvIndex, repeats = 5,
                                         classProbs = TRUE,verboseIter = FALSE,
                                         savePredictions = TRUE,
                                         summaryFunction = twoClassSummary) )

# knn classifiers 
knn_model <- train(output~ ., data = totcpr_3, method = "knn", metric="ROC",
                   #ceprePross = c("center","scale"),
                   #metric = "ROC", tuneLength = 10,
                   trControl=trainControl(method="repeatedcv", number=folds, 
                                          index = cvIndex, repeats = 5,
                                          classProbs = TRUE,verboseIter = FALSE,
                                          savePredictions = TRUE,
                                          summaryFunction = twoClassSummary) )

# CART
cart_model <- train(output~., data=totcpr_3, method="rpart", metric="ROC", 
                    trControl=trainControl(method="repeatedcv", number=folds, 
                                           index = cvIndex, repeats = 5,
                                           classProbs = TRUE, verboseIter = FALSE,
                                           savePredictions = TRUE,
                                           summaryFunction = twoClassSummary) )

# decision tree (c4.5)
totcpr_3$output <- as.factor(totcpr_3$output)
c45_model <- train(output ~.,data=totcpr_3, method="J48", metric="ROC",
                   tuneLength = 10,
                   trControl=trainControl(method="repeatedcv", number=folds, 
                                          index = cvIndex, classProbs = TRUE,
                                          savePredictions = TRUE, repeats = 5,
                                          summaryFunction = twoClassSummary) )

# SVM
svm_model <- train(output~., data = totcpr_3, method="svmRadial", metric="ROC",  
                   tuneLength = 10,
                   trControl=trainControl(method="cv", number=folds,
                                          index = cvIndex, classProbs = TRUE,
                                          savePredictions = TRUE, repeats = 5,
                                          summaryFunction = twoClassSummary) )

# 'SVM','CART','LogReg', 'NB', 'DTree(C4.5)','RF'
model_list <- list(svm = svm_model,
                   cart = cart_model,
                   glmmet = glm_model,
                   nb = nb_model,
                   c45 = c45_model,
                   rf = rf_model,
                   knn = knn_model)
resamp <- resamples(model_list)
resamp
summary(resamp)

# ROC curve
library(ROCR)
par(pin = c(4.0,4.0))
svm_pred <- prediction(svm_model$pred$X1, svm_model$pred$obs)
svm_perf <- performance(svm_pred,"tpr","fpr")
plot(svm_perf,lty=3, col="blue",main="ROC curve")
text(0.15, 0.89, labels="AUROC", cex=0.8)
text(0.15, 0.85, labels=sprintf("SVM: %0.4f", mean(svm_model$results$ROC)), col="blue", cex=0.8)
cart_pred <- prediction(cart_model$pred$X1, cart_model$pred$obs)
cart_perf <- performance(cart_pred,"tpr","fpr")
plot(cart_perf, col="red",add=TRUE)
text(0.15, 0.81, labels=sprintf("CART: %0.4f", mean(cart_model$results$ROC)), col="red", cex=0.8)
glm_pred <- prediction(glm_model$pred$X1, glm_model$pred$obs)
glm_perf <- performance(glm_pred,"tpr","fpr")
plot(glm_perf, col="purple",add=TRUE)
text(0.15, 0.77, labels=sprintf("LogReg: %0.4f", mean(glm_model$results$ROC)), col="purple", cex=0.8)
nb_pred <- prediction(nb_model$pred$X1, nb_model$pred$obs)
nb_perf <- performance(nb_pred,"tpr","fpr")
plot(nb_perf, col="brown",add=TRUE)
text(0.15, 0.73, labels=sprintf("NB: %0.4f", mean(nb_model$results$ROC)), col="brown", cex=0.8)
c45_pred <- prediction(c45_model$pred$X1, c45_model$pred$obs)
c45_perf <- performance(c45_pred,"tpr","fpr")
plot(c45_perf, col="black",add=TRUE)
text(0.15, 0.69, labels=sprintf("DTree(C4.5): %0.4f", mean(rf_model$results$ROC)), col="black", cex=0.8)
knn_pred <- prediction(knn_model$pred$X1, knn_model$pred$obs)
knn_perf <- performance(knn_pred,"tpr","fpr")
plot(knn_perf, col="rosybrown",add=TRUE)
text(0.15, 0.65, labels=sprintf("KNN: %0.4f", mean(knn_model$results$ROC)), col="rosybrown", cex=0.8)
rf_pred <- prediction(rf_model$pred$X1, rf_model$pred$obs)
rf_perf <- performance(rf_pred,"tpr","fpr")
plot(rf_perf, col="orange",add=TRUE)
text(0.15, 0.61, labels=sprintf("RF: %0.4f", mean(rf_model$results$ROC)), col="orange", cex=0.8)
legend(0.6,0.35,c('SVM','CART','Naive Bayes', 'DTree(C4.5)','Random forest','LogReg','knn'), 
       col = c('blue','red','brown', 'black','orange','brown','rosybrown'),cex = 0.7, lwd = 1)

#Pecision-Recall curve
library(DMwR)
par(pin = c(4.0,4.0))
PRcurve(svm_model$pred$X1, svm_model$pred$obs,lty=1, col="blue",main="Pecision-recall curve",
        xlim=c(0,1),ylim=c(0,1),avg="vertical")
PRcurve(cart_model$pred$X1, cart_model$pred$obs, add=TRUE,lty=1, col="red",avg="vertical")
PRcurve(glm_model$pred$X1, glm_model$pred$obs,add=TRUE, lty=1, col="purple",avg="vertical")
PRcurve(nb_model$pred$X1, nb_model$pred$obs,add=TRUE,lty=1,col="brown",avg="vertical")
PRcurve(c45_model$pred$X1, c45_model$pred$obs,add=TRUE,lty=1, col="black",avg="vertical")
PRcurve(knn_model$pred$X1, knn_model$pred$obs,add=TRUE,lty=1, col="rosybrown",avg="vertical")
PRcurve(rf_model$pred$X1, rf_model$pred$obs,add=TRUE,lty=1, col="orange",avg="vertical")
legend('bottomleft',c('SVM','CART','Naive Bayes', 'DTree(C4.5)','Random forest','LogReg','knn'),
       lty = c(1,1,1,1,1,1),col = c('blue','red','brown', 'black','orange','brown','rosybrown'))

# ROC for imbalanced classes
library(DMwR) 
library(purrr)
library(PRROC)
calc_auprc <- function(model, data)
{
  index_class2 <- data$output == "X1"
  index_class1 <- data$output == "X0"
  predictions <- predict(model, data, type = "prob")
  pr.curve(predictions$X1[index_class2], predictions$X1[index_class1], curve = TRUE)
}

model_list_pr <- model_list %>%
  map(calc_auprc, data = totcpr_3)

model_list_pr %>%
  map(function(the_mod) the_mod$auc.integral)

# Plot the AUPRC curve for all 7 models
results_list_pr <- list(NA)
num_mod <- 1
for(the_pr in model_list_pr){
  
  results_list_pr[[num_mod]] <- as.data.frame(recall = the_pr$curve[, 1],
                                              precision = the_pr$curve[, 2],
                                              model = names(model_list_pr)[num_mod])
  num_mod <- num_mod + 1
}
results_df_pr <- bind_rows(results_list_pr)
custom_col <- c("#000000", "#009E73", "#0072B2", "#D55E00", "#CC79A7","brown","red")
ggplot(aes(x = recall, y = precision, group = model), data = results_df_pr) +
  geom_line(aes(color = model), size = 1) +
  scale_color_manual(values = custom_col) +
  geom_abline(intercept = sum(totcpr_3$output == "X1")/nrow(totcpr_3),
              slope = 0, color = "gray", size = 1) +
  theme_bw()


# Plot a lift chart
#nb_perf_lc <- performance(nb_pred,"lift","rpp")
plot(performance(nb_pred,"lift","rpp"), main="Lift chart", col="brown")
plot(performance(svm_pred,"lift","rpp"), col="red",add=TRUE)

#Plot a Gain Chart
nb_perf_gc <- performance(nb_pred,"tpr","rpp")
plot(nb_perf_gc, main="Gain chart", col="blue",add=TRUE)


