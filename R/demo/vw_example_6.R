# Example for rvw
# Naive implementation for One Against All multi-class classification
# which is not yet available for rvw

# Implementation:
# 1) Train K (number of classes) binary classifiers one class against all others
# 2) Compute predictions for every classifier
# 3) Use prediction with maximum i-th class probability among classifiers to conduct final predictions


library(rvw)
library(data.table)
library(ggplot2)
library(mltools) # needed for some data preparetions
library(caret) # to compute confusion matrix

# Arbitrary but fixed seed
set.seed(123)


# Wrapper function to train models and compute predictions
wrapper <- function(dt, train_part = 0.8, target) {
  
  # Temporary directory to avoid leaving temp files
  cwd <- getwd()
  setwd(tempdir())
  
  # Extract levels
  dt[[target]] <- as.factor(dt[[target]])
  target_levels <- levels(dt[[target]])
  # Copy input data
  temp_dt <- data.table::copy(dt)
  
  # Prepare indices to split data
  ind_train <- sample(1:nrow(temp_dt), train_part*nrow(temp_dt))
  
  
  # Initialize containers for results
  res <- list(models = list(), preds_and_actual = data.table("actual" = as.factor(temp_dt[-ind_train,][[target]])))
  
  for (levels_elem in target_levels) {
    
    print(levels_elem)
    # Restore target variable to initial state and create {-1, 1} levels
    temp_dt[[target]] <- data.table::copy(dt[[target]])
    temp_dt[[target]] <- ifelse(temp_dt[[target]] == levels_elem, 1, -1)
    
    # Split data into train and validation subsets
    dt_train <- temp_dt[ind_train,]
    dt_val <- temp_dt[-ind_train,]
    
    # Direct use of data to train and predict using vw 
    res[["models"]][[levels_elem]] <- vw(training_data = dt_train,
                                                validation_data = dt_val,
                                                model = "mdl.vw",
                                                target = target,
                                                use_perf = FALSE,
                                                learning_rate = 0.1,
                                                loss = "logistic",
                                                link_function = "--link=logistic",
                                                passes = 10,
                                                plot_roc = TRUE,
                                                keep_tempfiles=FALSE,
                                                verbose = FALSE, 
                                                do_evaluation = FALSE)
    
    # Add actual data to plot roc later
    res[["models"]][[levels_elem]][["data"]][, actual:=as.factor(dt_val[[target]])]
    
    # Copy predictions to separate data.table for easier access
    invisible(res[["preds_and_actual"]][, (levels_elem) := res[["models"]][[levels_elem]][["data"]][,predicted]])
    
  }
  
  # Normalize predictions
  for (i in 1:nrow(res[["preds_and_actual"]])) {
    res[["preds_and_actual"]][i, 2:ncol(res[["preds_and_actual"]])] <-  res[["preds_and_actual"]][i, 2:ncol(res[["preds_and_actual"]])] / sum(res[["preds_and_actual"]][i, -1])
  }
  
  # Compute final predictions using simple max rule
  final_predictions <- apply(res[["preds_and_actual"]][,-1], 1, function(x) names(which.max(x)))
  res[["preds_and_actual"]][["predicted"]] <- as.factor(final_predictions)
  
  # Return back
  setwd(cwd)  
  
  return(res)
}


# We will use abalone dataset and will try to predict age groups (based on number of abalone shell rings) from physical measurements
aburl = 'http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
abnames = c('sex','length','diameter','height','weight.w','weight.s','weight.v','weight.sh','rings')
abalone = read.table(aburl, header = F , sep = ',', col.names = abnames)
data_full <- abalone

# Split number of rings into groups with equal (as possible) number of observations
data_full$group <- bin_data(data_full$rings, bins=3, binType = "quantile")
group_lvls <- levels(data_full$group)
levels(data_full$group) <- c(1, 2, 3)


# Run training with vw and then predict
resvw <- wrapper(dt = data_full, target = "group")

# Compute confusion matrix
confusionMatrix(resvw[["preds_and_actual"]][["predicted"]], resvw[["preds_and_actual"]][["actual"]])


# Plot ROC curves

roc_list <- list()
legend_vect <- c()
color_vect <- c()
for (i in 1:length(resvw[["models"]])) {
  print(i)
  # Rescale actual values to {0, 1}
  resvw[["models"]][[i]][["data"]][ actual=="-1", actual:="0" ]
  resvw[["models"]][[i]][["data"]][["actual"]] <- factor(resvw[["models"]][[i]][["data"]][["actual"]])
  roc_list[[i]] <- roc(resvw[["models"]][[i]][["data"]][["actual"]], resvw[["models"]][[i]][["data"]][["predicted"]])
  roc_list[[i]][["color"]] <- sample(colours(), 1)
  roc_list[[i]][["class_name"]] <- group_lvls[[i]]
  legend_vect <- c(legend_vect, paste(roc_list[[i]][["class_name"]], format(as.numeric(roc_list[[i]]$auc), digits=4)))
  color_vect <- c(color_vect, roc_list[[i]][["color"]])
  print(roc_list[[i]][["color"]])
  if (i == 1) {
    plot(roc_list[[i]], col= roc_list[[i]][["color"]], title = "ROC curves for individual classifiers OAA")
  } else {
    plot(roc_list[[i]], add=TRUE, col= roc_list[[i]][["color"]])
  }
}
# title(main ="ROC curves for individual classifiers OAA")
legend("bottomright",
       legend=legend_vect,
       col=color_vect, bty="n", lwd=3, title = "Group range for rings variable")

