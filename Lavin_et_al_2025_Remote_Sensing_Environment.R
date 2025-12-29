# Script for: High-resolution three-dimensional mapping of eelgrass
# (Zostera marina) habitat using drone-borne LiDAR 


# Charles P. Lavin 1, Toms Buls 2, Robert Noddebo Poulsen 2, Hege Gundersen 1, Kristina Oie Kvile 1, Cyvind Tangen Cdegaard 1, Kasper Hancke 1 

# 1 Norwegian Institute for Water Research (NIVA), Okernveien 94, 0579 Oslo, Norway 
# 2 SpectroFly ApS, Markstien 2, 4640 Faxe, Denmark 
# Corresponding author: charles.lavin@niva.no 

# Set encoding to UTF-8 to prevent encoding issues across different systems
Sys.setlocale("LC_ALL", "en_US.UTF-8")
options(encoding = "UTF-8")

# Helper function to fix CRS encoding issues
fix_crs_encoding <- function(las_object) {
  if (!is.null(las_object@crs$input)) {
    Encoding(las_object@crs$input) <- "UTF-8"
  }
  if (!is.null(las_object@crs$wkt)) {
    Encoding(las_object@crs$wkt) <- "UTF-8"
  }
  return(las_object)
}

# Load libraries
library(lidR)
library(RCSF)
library(rgl)
library(htmlwidgets)
library(gstat)
library(Metrics)
library(randomForest)
library(FNN)
library(caret)
library(sf)
library(sp)
library(terra)
library(RANN)
library(tidyverse)
library(stringr)
library(ggplot2)
library(ggnewscale)
library(ggpubr)
library(patchwork)
library(grid)
library(gridExtra)
library(tmap)
library(fields)


# Set working directory to the location of this R script
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))



# loading 50 m LiDAR data
las <- readLAS("North study area cleaned.laz")
las <- fix_crs_encoding(las)
summary(las) #ETRS89 / UTM zone 32N + NN2000 height
hist(las$Intensity) # pc density (square m)

summary(las$Intensity)


# pc density (voxel)
# point cloud density
grid_density <- grid_metrics(las, ~length(Z), res = 1)
summary(values(grid_density))

# cubic m density
vertical_resolution <- 1

# Bin Z values into discrete vertical slices and filter points
las@data$Z_bin <- floor(las@data$Z / vertical_resolution)
density_list <- lapply(unique(las@data$Z_bin), function(bin) {
  grid_metrics(filter_poi(las, Z_bin == bin), ~length(Z), res = 1)
})

# Calculate cell size (area) from the first raster
cell_area <- raster::res(density_list[[1]])[1] * raster::res(density_list[[1]])[2]

# Initialize total volume and points
total_volume <- 0
total_points <- 0

# Loop over each bin to compute volume and sum densities
for (density in density_list) {
  cell_counts <- sum(!is.na(values(density)))  # Count non-NA cells
  bin_points <- sum(values(density), na.rm = TRUE)  # Sum of points in this bin
  bin_volume <- cell_counts * cell_area * vertical_resolution  # Volume for this bin
  
  # Accumulate total volume and points
  total_volume <- total_volume + bin_volume
  total_points <- total_points + bin_points
}

# Calculate overall mean density
mean_density <- total_points / total_volume
# Print result
cat("Overall mean density (points per cubic meter):", mean_density, "\n")



# nearest neighbour distance
points <- cbind(las$X, las$Y, las$Z, las$Intensity)
knn_result <- knn.dist(points, k = 2)  # Get k-nearest neighbor distances
nn_distances <- knn_result[, 2]  # Extract nearest neighbor distance

# Combine the features into a data frame
data <- data.frame(
  Intensity = las$Intensity,
  NN_Distance = nn_distances
)

# Add the R, G, B values to the data frame
data$R <- las$R
data$G <- las$G
data$B <- las$B
data$Z <- las$Z

head(data)
summary(data$Intensity)




# Random forest model

# Annotate data set
# Loading ZOSMA GT height data
gt_points <- st_read("ZOSMA_GT_Height.shp")
crs_gt <- st_crs(gt_points)
Points <- c(23,29,30,31,32,34,35,37,38,39,40,41,42) # keep eelgrass points only
gt_points <- gt_points %>% filter(Point %in% Points)
gt_points <- gt_points %>% distinct(Point, .keep_all = TRUE)
gt_coords <- st_coordinates(gt_points)
st_crs(gt_points)

# Intensity values of eelgrass ground-truth points
points_df <- as.data.frame(points)
names(points_df) <- c("X", "Y", "Z", "Intensity")
nn_intensity <- nn2(data = points_df[, c("X", "Y")], query = gt_coords, k = 1)

# Extract indices of the nearest neighbors
nearest_intensity_indices <- nn_intensity$nn.idx[, 1]  # index of nearest neighbor for each gt_coord

# Extract the corresponding intensity values
nearest_intensity_50 <- points_df$Intensity[nearest_intensity_indices]

# Add intensity to your gt_points
gt_points$Nearest_Intensity <- nearest_intensity_50

summary(gt_points$Nearest_Intensity)
sd(gt_points$Nearest_Intensity)

# Mean intensity = 863
# S.D. = 400

data$classification <- ifelse(
  data$Intensity >= 0 & data$Intensity <= 1263,
  "vegetation", "sea_bottom"
)

# Convert annotation to a factor (for classification)
data$classification <- as.factor(data$classification)


# Add probability-based prior for RF model. Points within the range
# of ground-truth intensity values are marked as 90% likely to be
# vegetation, and points outside of this range are marked as 10%
# likely to be vegetation

data$vegetation_prob <- ifelse(
  data$Intensity <= 1263, 0.9, 0.1)



# Split the data into training and testing data sets (80% training, 20% testing)
set.seed(42)
train_index <- createDataPartition(data$Intensity, p = 0.8, list = FALSE)
training_data <- data[train_index, ]
testing_data <- data[-train_index, ]

# defining training control with random search
#set.seed(42)

# Define training control for random search
#control <- trainControl(
#  method = "cv",      # k-fold cross-validation
#  number = 5,         # 5-fold CV
#  search = "random"   # Perform random search
#)


# Define the number of random search iterations
#tune_length <- 10  # Number of random combinations to try

# Train the model using random search
#rf_model <- train(
#  classification ~ NN_Distance + vegetation_prob + R + G + B,
#  data = training_data,
#  method = "rf",
#  metric = "Accuracy",   # Optimize for accuracy
#  trControl = control,
#  tuneLength = tune_length  # Number of random parameter sets to try
#)

# Print the best hyperparameters found
#print(rf_model$bestTune)

# Make predictions
#test_predictions <- predict(rf_model, newdata = testing_data)

# Evaluate model performance
#conf_matrix <- confusionMatrix(test_predictions, testing_data$classification)
#print(conf_matrix)


# Train a Random Forest model with a large number of trees (e.g., 1000)
#set.seed(42)
#rf_large <- randomForest(
#  classification ~ NN_Distance + vegetation_prob + R + G + B,
#  data = training_data,
#  ntree = 500,  # Large number to observe convergence
#  importance = TRUE
#)

# Plot OOB error vs. number of trees
#plot(rf_large, main = "OOB Error vs. Number of Trees")


#tuneRF()
#set.seed(42)
#mtry_values <- 1:5  # Adjust based on the number of predictors
#oob_errors <- numeric(length(mtry_values))

#for (i in seq_along(mtry_values)) {
#  rf_model <- randomForest(
#    classification ~ NN_Distance + vegetation_prob + R + G + B,
#    data = training_data,
#    mtry = mtry_values[i],
#    ntree = 500
#  )
#  oob_errors[i] <- rf_model$err.rate[500, 1]  # OOB error at 500 trees
#}

# Find best mtry
#best_mtry <- mtry_values[which.min(oob_errors)]
#print(best_mtry)

#set.seed(42)
#tree_values <- seq(10, 100, by = 10)  # Try different ntree values
#oob_errors <- numeric(length(tree_values))

#for (i in seq_along(tree_values)) {
#  rf_model <- randomForest(
#    classification ~ NN_Distance + vegetation_prob + R + G + B,
#    data = training_data,
#    mtry = best_mtry,  # Use previously found best mtry
#    ntree = tree_values[i]
#  )
#  oob_errors[i] <- rf_model$err.rate[tree_values[i], 1]  # OOB error at final tree
#}

# Find the best ntree (min OOB error)
#best_ntree <- tree_values[which.min(oob_errors)]
#print(best_ntree)

# Plot OOB error vs ntree
#plot(tree_values, oob_errors, type = "b", pch = 19, col = "blue",
#     xlab = "Number of Trees", ylab = "OOB Error", main = "Choosing Optimal ntree")

# maxnodes
#set.seed(42)
#node_values <- seq(5, 50, by = 5)  # Try different maxnodes values
#oob_errors <- numeric(length(node_values))

#for (i in seq_along(node_values)) {
#  rf_model <- randomForest(
#    classification ~ NN_Distance + vegetation_prob + R + G + B,
#    data = training_data,
#    mtry = best_mtry,  # Optimal mtry from previous tests
#    ntree = best_ntree,  # Optimal ntree from previous tests
#    maxnodes = node_values[i]
#  )
#  oob_errors[i] <- rf_model$err.rate[best_ntree, 1]  # OOB error at final tree
#}

# Find the best maxnodes (min OOB error)
#best_maxnodes <- node_values[which.min(oob_errors)]
#print(best_maxnodes)

# Plot OOB error vs maxnodes
#plot(node_values, oob_errors, type = "b", pch = 19, col = "blue",
#     xlab = "Max Nodes", ylab = "OOB Error", main = "Choosing Optimal maxnodes")


# comparing with output from caret
# this defines mtry only
# Define the grid of hyperparameters for mtry only
#rf_grid <- expand.grid(
#  mtry = 1:5   # Test mtry values from 1 to 5
#)

# Set fixed values for ntree and maxnodes
#fixed_ntree <- 40   # Optimal ntree from previous tests
#fixed_maxnodes <- 50  # Optimal maxnodes from previous tests

# Train the model using caret with grid search for mtry only
#rf_model_caret <- train(
#  classification ~ NN_Distance + vegetation_prob + R + G + B,
#  data = training_data,
#  method = "rf",
#  metric = "Accuracy",  # Optimize for accuracy
#  trControl = control,  # Set cross-validation
#  tuneGrid = rf_grid,   # Specify the grid for mtry
#  ntree = fixed_ntree,  # Use previously determined best ntree
#  maxnodes = fixed_maxnodes  # Use previously determined best maxnodes
#)

# Print the best mtry found
#print(rf_model_caret$bestTune)

# Make predictions
#test_predictions <- predict(rf_model_caret, newdata = testing_data)

# Evaluate model performance
#conf_matrix <- confusionMatrix(test_predictions, testing_data$classification)
#print(conf_matrix)

# Compare results (manually tuned vs caret-tuned)
#print(paste("Manual mtry: ", best_mtry, " | Caret mtry: ", rf_model_caret$bestTune$mtry))
#print(paste("Manual ntree: ", best_ntree, " | Caret ntree: ", fixed_ntree))
#print(paste("Manual maxnodes: ", best_maxnodes, " | Caret maxnodes: ", fixed_maxnodes))





# Train the Random Forest model (with NN_Distance, vegetation_prob, R, G, B)
rf_model <- randomForest(
  classification ~ NN_Distance + vegetation_prob + R + G + B,  # Features
  data = training_data,
  ntree = 40,  # Optimal number of trees
  mtry = 2,
  importance = TRUE)  # Enable both MeanDecreaseAccuracy and MeanDecreaseGini

# Predict the classifications for the testing data
test_predictions <- predict(rf_model, newdata = testing_data)

# Add the predicted values to the testing data for comparison
testing_data$predicted_classification <- test_predictions

#  Compare the predicted values with the true classifications (confusion matrix)
conf_matrix <- confusionMatrix(testing_data$predicted_classification, testing_data$classification)

# Print the confusion matrix
print(conf_matrix)

# RF model performance
varImpPlot(rf_model)
plot(rf_model)
# Get the OOB error rate
rf_model$err.rate

# Error data structure
rf_error_data <- as.data.frame(rf_model$err.rate)
str(rf_model$err.rate)



# For ggplot

# 1. Extract the error rate data from the Random Forest model
rf_error_data <- rf_error_data %>%
  mutate(Trees = seq_len(nrow(rf_error_data))) %>%
  pivot_longer(
    cols = -Trees,
    names_to = "Error_Type",
    values_to = "Error_Rate"
  )

unique(rf_error_data$Error_Type)

rf_error_data %>% filter(is.na(Error_Rate))

rf_error_data <- rf_error_data %>%
  group_by(Error_Type) %>%
  mutate(Error_Rate = zoo::na.approx(Error_Rate, na.rm = FALSE, rule = 2)) %>%
  ungroup()

rf_error_data %>% summarize(total_na = sum(is.na(Error_Rate)))

rf_error_data <- rf_error_data %>%
  arrange(Error_Type, Trees)

ggplot(rf_error_data %>% filter(Error_Type == "vegetation"), 
       aes(x = Trees, y = Error_Rate)) +
  geom_line() +
  labs(title = "OOB Error Rate", x = "Number of Trees", y = "Error Rate") +
  theme_minimal()


S_Figure_3A <- ggplot(rf_error_data, aes(x = Trees, y = Error_Rate, color = Error_Type, linetype = Error_Type)) +
  geom_path(linewidth = 1) +
  labs(title = "50 m LiDAR",
       x = "Number of Trees",
       y = "Error Rate",
       color = "Error Type",
       linetype = "Error Type"
  ) +
  scale_color_manual(
    values = c("OOB" = "black", "sea_bottom" = "orange", "vegetation" = "purple"),
    labels = c("OOB" = "OOB Error", "sea_bottom" = "Sea floor error", "vegetation" = "Vegetation error")
  ) +
  scale_linetype_manual(
    values = c("OOB" = "solid", "sea_bottom" = "dashed", "vegetation" = "dotted"),
    labels = c("OOB" = "OOB Error", "sea_bottom" = "Sea floor error", "vegetation" = "Vegetation error")
  ) +
  theme_classic() +
  theme(
    plot.title = element_text(size=20),
    axis.text = element_text(size = 18),
    axis.title = element_text(size = 20),
    legend.text = element_text(size = 12),
    legend.title = element_text(size = 14),
    legend.position = c(0.8, 0.7))

S_Figure_3A


# 2. Extract variable importance data from the Random Forest model
# Using MeanDecreaseAccuracy (permutation importance) - more robust than MeanDecreaseGini
importance_data <- data.frame(
  Variable = rownames(rf_model$importance),
  MeanDecreaseAccuracy = rf_model$importance[, "MeanDecreaseAccuracy"],  # Permutation importance
  MeanDecreaseGini = rf_model$importance[, "MeanDecreaseGini"]  # Keep for comparison
)

# Plot the variable importance using MeanDecreaseAccuracy
S_Figure_3C <- ggplot(importance_data, aes(x = reorder(Variable, MeanDecreaseAccuracy), y = MeanDecreaseAccuracy)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "50 m LiDAR", x = "Variable", y = "Mean Decrease Accuracy") +
  theme_classic() +
  theme(
    plot.title = element_text(size=20),
    axis.text = element_text(size = 18),
    axis.title = element_text(size = 20),
    legend.text = element_text(size = 12),
    legend.title = element_text(size = 14)) +
  scale_x_discrete(labels = function(x) {
    x <- gsub("vegetation_prob", "Probability-based prior", x)
    x <- gsub("NN_Distance", "N.N. Distance", x)
    return(x)})

S_Figure_3C


# Now, predict classification on entire dataset from RF model
data$predicted_classification <- predict(rf_model, newdata = data)
table(is.na(data$predicted_classification))


# performing k-fold cross validation
#control <- trainControl(method = "cv", number = 10)
#rf_cv <- train(
#  classification ~ NN_Distance + vegetation_prob + R + G + B,
#  data = training_data,
#  method = "rf",
#  trControl = control,
#  tuneGrid = expand.grid(mtry = c(2, 3, 4))  # Varying mtry
#)
#print(rf_cv)



# Return classified points to las
las <- add_attribute(las, data$predicted_classification, "predicted_classification")

# Verify the LAS object
print(las)
print(header(las))

# verify results
table(is.na(data$predicted_classification)) 

plot(las)

# check classifications
summary(las$predicted_classification)

# Convert predicted_classification to numeric: 2 for sea bottom, 42 for vegetation
las$predicted_classification <- ifelse(las$predicted_classification == "sea_bottom", 2, 
                                       ifelse(las$predicted_classification == "vegetation", 42, NA))

summary(las$predicted_classification)

# Filter points where predicted_classification == 2 (sea bottom)
sea_bottom_las <- filter_poi(las, predicted_classification == 2)

plot(sea_bottom_las, color = "predicted_classification", main = "sea bottom Points")

# Filter points where predicted_classification == 42 (vegetation)
vegetation_las <- filter_poi(las, predicted_classification == 42)

# Plot the sea bottom points
plot(vegetation_las, color = "predicted_classification", main = "vegetation Points")

summary(las$predicted_classification)

# Assign colors based on predicted_classification
#plot(las, color = "predicted_classification", main = "All Points: Vegetation and Sea Bottom")

# save vegetation and sea bottom classes
#writeLAS(vegetation_las, "vegetation_points_from_RF.laz")
#writeLAS(sea_bottom_las, "sea_bottom_points_from_RF.laz")

# PC summaries
summary(sea_bottom_las)
summary(vegetation_las)



# merged classified points
classified_merged <- rbind(sea_bottom_las, vegetation_las)
summary(classified_merged)


##################################
# 25 m dataset
las_25 <- readLAS("North study area 25m updated.las")
las_25 <- fix_crs_encoding(las_25)
summary(las_25) #ETRS89 / UTM zone 32N + NN2000 height
hist(las_25$Intensity) # pc density (square m)

# pc density (voxel)
# point cloud density
grid_density <- grid_metrics(las_25, ~length(Z), res = 1)
summary(values(grid_density))

# cubic m density
vertical_resolution <- 1

# Bin Z values into discrete vertical slices and filter points
las_25@data$Z_bin <- floor(las_25@data$Z / vertical_resolution)
density_list <- lapply(unique(las_25@data$Z_bin), function(bin) {
  grid_metrics(filter_poi(las_25, Z_bin == bin), ~length(Z), res = 1)
})

# Calculate cell size (area) from the first raster
cell_area <- raster::res(density_list[[1]])[1] * raster::res(density_list[[1]])[2]

# Initialize total volume and points
total_volume <- 0
total_points <- 0

# Loop over each bin to compute volume and sum densities
for (density in density_list) {
  cell_counts <- sum(!is.na(values(density)))  # Count non-NA cells
  bin_points <- sum(values(density), na.rm = TRUE)  # Sum of points in this bin
  bin_volume <- cell_counts * cell_area * vertical_resolution  # Volume for this bin
  
  # Accumulate total volume and points
  total_volume <- total_volume + bin_volume
  total_points <- total_points + bin_points
}

# Calculate overall mean density
mean_density <- total_points / total_volume

# Print result
cat("Overall mean density (points per cubic meter):", mean_density, "\n")



# nearest neighbour distance
points <- cbind(las_25$X, las_25$Y, las_25$Z, las_25$Intensity)
knn_result <- knn.dist(points, k = 2)  # Get k-nearest neighbor distances
nn_distances <- knn_result[, 2]  # Extract nearest neighbor distance

# Combine the features into a data frame
data <- data.frame(
  Intensity = las_25$Intensity,
  NN_Distance = nn_distances
)

# Add the R, G, B values to the data frame
data$R <- las_25$R
data$G <- las_25$G
data$B <- las_25$B
data$Z <- las_25$Z

head(data)
summary(data$Intensity)




# RF

# Annotate data set

# Load ZOSMA ground truth points
# Loading ZOSMA GT height data
gt_points <- st_read("ZOSMA_GT_Height.shp")
crs_gt <- st_crs(gt_points)
Points <- c(23,29,30,31,32,34,35,37,38,39,40,41,42) # eelgrass points only
gt_points <- gt_points %>% filter(Point %in% Points)
gt_points <- gt_points %>% distinct(Point, .keep_all = TRUE)
gt_coords <- st_coordinates(gt_points)
st_crs(gt_points)

# Intensity values
points_df <- as.data.frame(points)
names(points_df) <- c("X", "Y", "Z", "Intensity")
nn_intensity <- nn2(data = points_df[, c("X", "Y")], query = gt_coords, k = 1)

# Extract indices of the nearest neighbors
nearest_intensity_indices <- nn_intensity$nn.idx[, 1]  # index of nearest neighbor for each gt_coord

# Extract the corresponding intensity values
nearest_intensity_50 <- points_df$Intensity[nearest_intensity_indices]

# Add intensity to your gt_points
gt_points$Nearest_Intensity <- nearest_intensity_50

summary(gt_points$Nearest_Intensity)
sd(gt_points$Nearest_Intensity)

# Mean = 1512
# S.D. = 298


data$classification <- ifelse(
  data$Intensity >= 0 & data$Intensity <= 1810,
  "vegetation", "sea_bottom"
)

# Convert annotation to a factor (for classification)
data$classification <- as.factor(data$classification)


# Add probability-based prior for RF model. Points within the range
# of ground-truth intensity values are marked as 90% likely to be
# vegetation, and points outside of this range are marked as 10%
# likely to be vegetation

data$vegetation_prob <- ifelse(
  data$Intensity <= 1810, 0.9, 0.1)



# Split the data into training and testing data sets (80% training, 20% testing)
set.seed(42)
train_index <- createDataPartition(data$Intensity, p = 0.8, list = FALSE)
training_data <- data[train_index, ]
testing_data <- data[-train_index, ]




##########
# defining training control with random search
#set.seed(42)

# Define training control for random search
#control <- trainControl(
#  method = "cv",      # k-fold cross-validation
#  number = 5,         # 5-fold CV
#  search = "random"   # Perform random search
#)


# Define the number of random search iterations
#tune_length <- 10  # Number of random combinations to try

# Train the model using random search
#rf_model <- train(
#  classification ~ NN_Distance + vegetation_prob + R + G + B,
#  data = training_data,
#  method = "rf",
#  metric = "Accuracy",
#  trControl = control,
#  tuneLength = tune_length
#)

# Print the best hyperparameters found
#print(rf_model$bestTune)

# Make predictions
#test_predictions <- predict(rf_model, newdata = testing_data)

# Evaluate model performance
#conf_matrix <- confusionMatrix(test_predictions, testing_data$classification)
#print(conf_matrix)


# Update, 07-02-25
# Train a Random Forest model with a large number of trees (e.g., 1000)
#set.seed(42)
#library(ranger)

#rf_large <- ranger(
#  classification ~ NN_Distance + vegetation_prob + R + G + B,
#  data = training_data,
#  num.trees = 500,
#  importance = "impurity",
#  probability = TRUE
#)

#rf_large


# Define grid of hyperparameters
#tune_grid <- expand.grid(
#  mtry = c(2, 3, 4, 5),  # Try different numbers of variables at each split
#  splitrule = "gini",
#  min.node.size = c(5, 10, 20)  # Test different node sizes
#)

# Define training control
#control <- trainControl(
#  method = "cv",  # Cross-validation
#  number = 5,     # 5-fold CV
#  search = "grid"
#)

# Train the model
#set.seed(42)
#rf_tuned <- train(
#  classification ~ NN_Distance + vegetation_prob + R + G + B,
#  data = training_data,
#  method = "ranger",
#  metric = "Accuracy",
#  trControl = control,
#  tuneGrid = tune_grid,
#  num.trees = 500
#)

# Print best parameters
#print(rf_tuned$bestTune)




# Plot OOB error vs. number of trees
#plot(rf_large, main = "OOB Error vs. Number of Trees")

#tuneRF()
#set.seed(42)
#mtry_values <- 1:3  # Adjust based on the number of predictors
#oob_errors <- numeric(length(mtry_values))

#for (i in seq_along(mtry_values)) {
#  rf_model <- randomForest(
#    classification ~ NN_Distance + vegetation_prob + R + G + B,
#    data = training_data,
#    mtry = mtry_values[i],
#    ntree = 40
#  )
#  oob_errors[i] <- rf_model$err.rate[40, 1]  # OOB error at 500 trees
#}

# Find best mtry
#best_mtry <- mtry_values[which.min(oob_errors)]
#print(best_mtry)

#set.seed(42)
#tree_values <- seq(10, 100, by = 10)  # Try different ntree values
#oob_errors <- numeric(length(tree_values))

#for (i in seq_along(tree_values)) {
#  rf_model <- randomForest(
#    classification ~ NN_Distance + vegetation_prob + R + G + B,
#    data = training_data,
#    mtry = best_mtry,  # Use previously found best mtry
#    ntree = tree_values[i]
#  )
#  oob_errors[i] <- rf_model$err.rate[tree_values[i], 1]  # OOB error at final tree
#}

# Find the best ntree (min OOB error)
#best_ntree <- tree_values[which.min(oob_errors)]
#print(best_ntree)

# Plot OOB error vs ntree
#plot(tree_values, oob_errors, type = "b", pch = 19, col = "blue",
#     xlab = "Number of Trees", ylab = "OOB Error", main = "Choosing Optimal ntree")

# maxnodes
#set.seed(42)
#node_values <- seq(5, 50, by = 5)  # Try different maxnodes values
#oob_errors <- numeric(length(node_values))

#for (i in seq_along(node_values)) {
#  rf_model <- randomForest(
#    classification ~ NN_Distance + vegetation_prob + R + G + B,
#    data = training_data,
#    mtry = best_mtry,  # Optimal mtry from previous tests
#    ntree = best_ntree,  # Optimal ntree from previous tests
#    maxnodes = node_values[i]
#  )
#  oob_errors[i] <- rf_model$err.rate[best_ntree, 1]  # OOB error at final tree
#}

# Find the best maxnodes (min OOB error)
#best_maxnodes <- node_values[which.min(oob_errors)]
#print(best_maxnodes)

# Plot OOB error vs maxnodes
#plot(node_values, oob_errors, type = "b", pch = 19, col = "blue",
#     xlab = "Max Nodes", ylab = "OOB Error", main = "Choosing Optimal maxnodes")


# comparing with output from caret
# this defines mtry only
# Define the grid of hyperparameters for mtry only
#rf_grid <- expand.grid(
#  mtry = 1:5   # Test mtry values from 1 to 5
#)

# Set fixed values for ntree and maxnodes
#fixed_ntree <- 40   # Optimal ntree from previous tests
#fixed_maxnodes <- 50  # Optimal maxnodes from previous tests

# Train the model using caret with grid search for mtry only
#rf_model_caret <- train(
#  classification ~ NN_Distance + vegetation_prob + R + G + B,
#  data = training_data,
#  method = "rf",
#  metric = "Accuracy",  # Optimize for accuracy
#  trControl = control,  # Set cross-validation
#  tuneGrid = rf_grid,   # Specify the grid for mtry
#  ntree = fixed_ntree,  # Use previously determined best ntree
#  maxnodes = fixed_maxnodes  # Use previously determined best maxnodes
#)

# Print the best mtry found
#print(rf_model_caret$bestTune)

# Make predictions
#test_predictions <- predict(rf_model_caret, newdata = testing_data)

# Evaluate model performance
#conf_matrix <- confusionMatrix(test_predictions, testing_data$classification)
#print(conf_matrix)

# Compare results (manually tuned vs caret-tuned)
#print(paste("Manual mtry: ", best_mtry, " | Caret mtry: ", rf_model_caret$bestTune$mtry))
#print(paste("Manual ntree: ", best_ntree, " | Caret ntree: ", fixed_ntree))
#print(paste("Manual maxnodes: ", best_maxnodes, " | Caret maxnodes: ", fixed_maxnodes))





##########
# Train the Random Forest model (with NN_Distance, vegetation_prob, R, G, B)
rf_model <- randomForest(
  classification ~ NN_Distance + vegetation_prob + R + G + B,  # Features
  data = training_data,
  ntree = 40,  # Optimal number of trees
  mtry = 1,
  importance = TRUE)  # Enable both MeanDecreaseAccuracy and MeanDecreaseGini

# Predict the classifications for the testing data
test_predictions <- predict(rf_model, newdata = testing_data)

# Add the predicted values to the testing data for comparison
testing_data$predicted_classification <- test_predictions

#  Compare the predicted values with the true classifications (confusion matrix)
conf_matrix <- confusionMatrix(testing_data$predicted_classification, testing_data$classification)

# Print the confusion matrix
print(conf_matrix)

# RF model performance
varImpPlot(rf_model)
plot(rf_model)
# Get the OOB error rate
rf_model$err.rate

# Error data structure
rf_error_data <- as.data.frame(rf_model$err.rate)
str(rf_model$err.rate)



# For ggplot

# 1. Extract the error rate data from the Random Forest model
rf_error_data <- rf_error_data %>%
  mutate(Trees = seq_len(nrow(rf_error_data))) %>%
  pivot_longer(
    cols = -Trees,
    names_to = "Error_Type",
    values_to = "Error_Rate"
  )

unique(rf_error_data$Error_Type)

rf_error_data %>% filter(is.na(Error_Rate))

rf_error_data <- rf_error_data %>%
  group_by(Error_Type) %>%
  mutate(Error_Rate = zoo::na.approx(Error_Rate, na.rm = FALSE, rule = 2)) %>%
  ungroup()

rf_error_data %>% summarize(total_na = sum(is.na(Error_Rate)))

rf_error_data <- rf_error_data %>%
  arrange(Error_Type, Trees)

ggplot(rf_error_data %>% filter(Error_Type == "vegetation"), 
       aes(x = Trees, y = Error_Rate)) +
  geom_line() +
  labs(title = "OOB Error Rate", x = "Number of Trees", y = "Error Rate") +
  theme_minimal()


S_Figure_3B <- ggplot(rf_error_data, aes(x = Trees, y = Error_Rate, color = Error_Type, linetype = Error_Type)) +
  geom_path(linewidth = 1) +
  labs(title = "25 m LiDAR",
       x = "Number of Trees",
       y = "",
       color = "Error Type",
       linetype = "Error Type"
  ) +
  scale_color_manual(
    values = c("OOB" = "black", "sea_bottom" = "orange", "vegetation" = "purple"),
    labels = c("OOB" = "OOB Error", "sea_bottom" = "Sea floor error", "vegetation" = "Vegetation error")
  ) +
  scale_linetype_manual(
    values = c("OOB" = "solid", "sea_bottom" = "dashed", "vegetation" = "dotted"),
    labels = c("OOB" = "OOB Error", "sea_bottom" = "Sea floor error", "vegetation" = "Vegetation error")
  ) +
  theme_classic() +
  theme(
    plot.title = element_text(size=20),
    axis.text = element_text(size = 18),
    axis.title = element_text(size = 20),
    legend.text = element_text(size = 12),
    legend.title = element_text(size = 14),
    legend.position = c(0.8, 0.7))

S_Figure_3B







# 2. Extract variable importance data from the Random Forest model
# Using MeanDecreaseAccuracy (permutation importance) - more robust than MeanDecreaseGini
importance_data <- data.frame(
  Variable = rownames(rf_model$importance),
  MeanDecreaseAccuracy = rf_model$importance[, "MeanDecreaseAccuracy"],  # Permutation importance
  MeanDecreaseGini = rf_model$importance[, "MeanDecreaseGini"]  # Keep for comparison
)

# Plot the variable importance using MeanDecreaseAccuracy
S_Figure_3D <- ggplot(importance_data, aes(x = reorder(Variable, MeanDecreaseAccuracy), y = MeanDecreaseAccuracy)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "25 m LiDAR", x = "", y = "Mean Decrease Accuracy") +
  theme_classic() +
  theme(
    plot.title = element_text(size = 20),
    axis.text = element_text(size = 18),
    axis.title = element_text(size = 20),
    legend.text = element_text(size = 12),
    legend.title = element_text(size = 14)) +
  scale_x_discrete(labels = function(x) {
    x <- gsub("vegetation_prob", "Probability-based prior", x)
    x <- gsub("NN_Distance", "N.N. Distance", x)
    return(x)})

S_Figure_3D

# S Figure 2 
S_Figure_3 <- ggarrange(S_Figure_3A, S_Figure_3B,
                        S_Figure_3C, S_Figure_3D,
                        ncol=2, nrow=2,
                        align = c("hv"),
                        labels = c("A", "B", "C", "D"),
                        hjust = -6,
                        font.label = list(size = 20))

S_Figure_3

# Save the arranged figure as a .tif file
#ggsave("S_Figure_3_new.png", plot = S_Figure_3, device = "png", width = 15, height = 8, dpi = 300)









# Now, predict classification on entire dataset from RF model
data$predicted_classification <- predict(rf_model, newdata = data)
table(is.na(data$predicted_classification))


# performing k-fold cross validation
#control <- trainControl(method = "cv", number = 10)
#rf_cv <- train(
#  classification ~ NN_Distance + vegetation_prob + R + G + B,
#  data = training_data,
#  method = "rf",
#  trControl = control,
#  tuneGrid = expand.grid(mtry = c(2, 3, 4))  # Varying mtry
#)
#print(rf_cv)



# Return classified points to las

las_25 <- add_attribute(las_25, data$predicted_classification, "predicted_classification")

# Verify the LAS object
print(las_25)
print(header(las_25))

# verify results
table(is.na(data$predicted_classification)) 

plot(las_25)

# check classifications
summary(las_25$predicted_classification)

# Convert predicted_classification to numeric: 2 for sea bottom, 42 for vegetation
las_25$predicted_classification <- ifelse(las_25$predicted_classification == "sea_bottom", 2, 
                                          ifelse(las_25$predicted_classification == "vegetation", 42, NA))

summary(las_25$predicted_classification)

# Filter points where predicted_classification == 2 (sea bottom)
sea_bottom_las_25 <- filter_poi(las_25, predicted_classification == 2)

plot(sea_bottom_las_25, color = "predicted_classification", main = "sea bottom Points")

# Filter points where predicted_classification == 42 (vegetation)
vegetation_las_25 <- filter_poi(las_25, predicted_classification == 42)

# Plot the sea bottom points
plot(vegetation_las_25, color = "predicted_classification", main = "vegetation Points")

summary(las_25$predicted_classification)

# Assign colors based on predicted_classification
#plot(las, color = "predicted_classification", main = "All Points: Vegetation and Sea Bottom")

# save vegetation and sea bottom classes
#writeLAS(vegetation_las_25_25, "vegetation_points_from_RF.laz")
#writeLAS(sea_bottom_las_25, "sea_bottom_points_from_RF.laz")

# PC summaries
summary(sea_bottom_las_25)
summary(vegetation_las_25)



# merged classified points
classified_merged_25 <- rbind(sea_bottom_las_25, vegetation_las_25)
summary(classified_merged_25)



########################

# Creating DTM from classified points


# 50 m
# checking the average distance between points for both datasets
# Extract coordinates (X, Y, Z) as a matrix for sea_bottom_las
sea_bottom_points <- cbind(sea_bottom_las@data$X, sea_bottom_las@data$Y, sea_bottom_las@data$Z)

# Extract coordinates (X, Y, Z) as a matrix for vegetation_las
vegetation_points <- cbind(vegetation_las@data$X, vegetation_las@data$Y, vegetation_las@data$Z, vegetation_las@data$Intensity)
vegetation_points_area <- cbind(vegetation_las@data$X, vegetation_las@data$Y, vegetation_las@data$Z)


# all points joined together
study_area_points <- rbind(sea_bottom_points, vegetation_points_area)

# Compute nearest neighbor distances
knn_area <- knn.dist(study_area_points, k = 2)  # k=2 for nearest neighbor
nn_distances_area <- knn_area[, 2]

# Compute the average distance
avg_distance_area <- mean(nn_distances_area, na.rm = TRUE)
cat("Average spacing between points in area_las (50 m): ", avg_distance_area, "meters\n")

# SD
n <- sum(!is.na(nn_distances_area))
# Compute the standard deviation
sd_nn_distances <- sd(nn_distances_area, na.rm = TRUE)

# Compute standard error
se_nn_distances <- sd_nn_distances / sqrt(n)
cat("SE: ", se_nn_distances, "meters\n")


# Compute nearest neighbor distances for sea_bottom_las
knn_sea_bottom <- knn.dist(sea_bottom_points, k = 2)
nn_distances_sea_bottom <- knn_sea_bottom[, 2]

# Compute the average distance for sea_bottom_las
avg_distance_sea_bottom <- mean(nn_distances_sea_bottom, na.rm = TRUE)
cat("Average spacing between points in sea_bottom_las: ", avg_distance_sea_bottom, "meters\n")

# SE
n <- sum(!is.na(nn_distances_sea_bottom))
# Compute the standard deviation
sd_nn_distances <- sd(nn_distances_sea_bottom, na.rm = TRUE)
# Compute standard error
se_nn_distances <- sd_nn_distances / sqrt(n)
cat("SE: ", se_nn_distances, "meters\n")


# Extract coordinates (X, Y, Z) as a matrix for vegetation_las
vegetation_points <- cbind(vegetation_las@data$X, vegetation_las@data$Y, vegetation_las@data$Z, vegetation_las@data$Intensity)
vegetation_points_area <- cbind(vegetation_las@data$X, vegetation_las@data$Y, vegetation_las@data$Z)

# Compute nearest neighbor distances for vegetation_las
knn_vegetation <- knn.dist(vegetation_points, k = 2)
nn_distances_vegetation <- knn_vegetation[, 2]

# Compute the average distance for vegetation_las
avg_distance_vegetation <- mean(nn_distances_vegetation, na.rm = TRUE)
cat("Average spacing between points in vegetation_las: ", avg_distance_vegetation, "meters\n")
# vegetation points are patchily distributed across substrate, makes sense

# SE
n <- sum(!is.na(nn_distances_vegetation))
# Compute the standard deviation
sd_nn_distances <- sd(nn_distances_vegetation, na.rm = TRUE)
# Compute standard error
se_nn_distances <- sd_nn_distances / sqrt(n)
cat("SE: ", se_nn_distances, "meters\n")

# Vegetation intensity summary
veg_info <- as.data.frame(vegetation_points)
summary(veg_info)


# Creating sea bottom DTMs (50 m)
# First, we need to classify sea bottom points as ground points in lidR package
sea_bottom_las_ground <- classify_ground(sea_bottom_las, algorithm = csf())
# check to see if it worked
gnd <- filter_ground(sea_bottom_las_ground)
plot(gnd, size = 3, bg = "white") 

# summary of ground points
summary(gnd)

plot(sea_bottom_las_ground)
plot(gnd)

# check gnd
str(gnd)
class(gnd)

#################################
# Fix UTF-8 encoding issues
gnd <- fix_crs_encoding(gnd)
#################################

# DTM, IDW
dtm_idw <- rasterize_terrain(gnd, res = 0.1, algorithm = knnidw(k = 10L, p = 2))
#Plot IDW with PC
plot(vegetation_las) |> add_dtm3d(dtm_idw)

# Compare LiDAR canopy height with ground truth canopy height
# Convert vegetation LAS point cloud into a dataframe for NN analysis
veg_df <- data.frame(
  X = vegetation_las@data$X,
  Y = vegetation_las@data$Y,
  Z = vegetation_las@data$Z
)

nn <- nn2(data = veg_df[, c("X", "Y")], query = gt_coords, k = 5)

# Extract Z-values for the nearest neighbors
neighbor_indices <- nn$nn.idx
z_values <- apply(neighbor_indices, 1, function(indices) veg_df$Z[indices])

# Calculate the Z range for each GT point
z_min <- apply(z_values, 2, min)
z_max <- apply(z_values, 2, max)
z_mean <- apply(z_values, 2, mean)

# Add Z-values to GT point data
gt_points$Z_min_veg <- z_min
gt_points$Z_max_veg <- z_max
gt_points$Z_mean_veg <- z_mean


# Smoothing raster
idw_smoothed <- focal(dtm_idw, w=87, fun=mean, na.rm=FALSE)
gt_points <- st_transform(gt_points, crs(idw_smoothed))

# extract ground truth elevation, subtract from canopy
gt_points$idw_smoothed_bathy <- terra::extract(idw_smoothed, gt_points)
gt_points$IDW_LiDAR_Canopy_smoothed <- gt_points$Z_max_veg - gt_points$idw_smoothed_bathy




###################
# 25 m DTM
# Extract coordinates (X, Y, Z) as a matrix for sea_bottom_las_25
sea_bottom_points_25 <- cbind(sea_bottom_las_25@data$X, sea_bottom_las_25@data$Y, sea_bottom_las_25@data$Z)

# Extract coordinates (X, Y, Z) as a matrix for vegetation_las_25
vegetation_points_25 <- cbind(vegetation_las_25@data$X, vegetation_las_25@data$Y, vegetation_las_25@data$Z, vegetation_las_25@data$Intensity)
vegetation_points_25_area <- cbind(vegetation_las_25@data$X, vegetation_las_25@data$Y, vegetation_las_25@data$Z)


# all points joined together
study_area_points_25 <- rbind(sea_bottom_points_25, vegetation_points_25_area)

# Compute nearest neighbor distances
knn_area <- knn.dist(study_area_points_25, k = 2)  # k=2 for nearest neighbor
nn_distances_area <- knn_area[, 2]

# Compute the average distance
avg_distance_area <- mean(nn_distances_area, na.rm = TRUE)
cat("Average spacing between points in area_las: ", avg_distance_area, "meters\n")

# Compute the sd
sd_nn_distances <- sd(nn_distances_area, na.rm = TRUE)
cat("SD of mean distance between points in area_las: ", sd_nn_distances, "meters\n")

# Compute standard error
se_nn_distances <- sd_nn_distances / sqrt(sum(!is.na(nn_distances_area)))
cat("Standard Error of nearest neighbor distances: ", se_nn_distances, "meters\n")


# Compute nearest neighbor distances for sea_bottom_las_25
knn_sea_bottom <- knn.dist(sea_bottom_points_25, k = 2)
nn_distances_sea_bottom <- knn_sea_bottom[, 2]

# Compute the average distance for sea_bottom_las_25
avg_distance_sea_bottom <- mean(nn_distances_sea_bottom, na.rm = TRUE)
cat("Average spacing between points in sea_bottom_las_25: ", avg_distance_sea_bottom, "meters\n")



# Compute nearest neighbor distances for vegetation_las_25
knn_vegetation <- knn.dist(vegetation_points_25, k = 2)
nn_distances_vegetation <- knn_vegetation[, 2]

# Compute the average distance for vegetation_las_25
avg_distance_vegetation <- mean(nn_distances_vegetation, na.rm = TRUE)
cat("Average spacing between points in vegetation_las_25: ", avg_distance_vegetation, "meters\n")
# vegetation points are patchily distributed across substrate, makes sense

# crs of las datasets
crs(sea_bottom_las_25)

# bottom depth summary
depth_info <- as.data.frame(sea_bottom_points_25)
summary(depth_info)

# Intensity summary
veg_info <- as.data.frame(vegetation_points_25)
summary(veg_info) #V4 is intensity


# First, we need to classify sea bottom points as ground points in lidR package
sea_bottom_las_25_ground <- classify_ground(sea_bottom_las_25, algorithm = csf())
# check to see if it worked
gnd_25 <- filter_ground(sea_bottom_las_25_ground)
plot(gnd_25, size = 3, bg = "white") 


# summary of ground points
summary(gnd_25)

#################################
# Fix UTF-8 encoding issues
gnd_25 <- fix_crs_encoding(gnd_25)
################################

# DTM, IDW
dtm_idw_25 <- rasterize_terrain(gnd_25, res = 0.1, algorithm = knnidw(k = 10L, p = 2))
plot(vegetation_las_25) |> add_dtm3d(dtm_idw_25)



# Now, comparing GT canopy height, to that from our new DTM models
# Convert vegetation LAS point cloud into a dataframe for NN analysis
veg_df_25 <- data.frame(
  X = vegetation_las_25@data$X,
  Y = vegetation_las_25@data$Y,
  Z = vegetation_las_25@data$Z
)

nn <- nn2(data = veg_df_25[, c("X", "Y")], query = gt_coords, k = 5)

# Extract Z-values for the nearest neighbors
neighbor_indices <- nn$nn.idx
z_values <- apply(neighbor_indices, 1, function(indices) veg_df_25$Z[indices])

# Calculate the Z range for each GT point
z_min <- apply(z_values, 2, min)
z_max <- apply(z_values, 2, max)
z_mean <- apply(z_values, 2, mean)

# Add Z-values to GT point data
gt_points$Z_min_veg_25 <- z_min
gt_points$Z_max_veg_25 <- z_max
gt_points$Z_mean_veg_25 <- z_mean

# SMOOTHING rasters
# Apply the filter to the entire DTM, w = 87
idw_smoothed_25 <- focal(dtm_idw_25, w=87, fun=mean, na.rm=FALSE)
plot_dtm3d(dtm_idw_25, bg = "white") 
plot_dtm3d(idw_smoothed_25, bg = "white") 

# Calculate ZOSMA canopoy height from smoothed rasters
gt_points$idw_smoothed_25_bathy <- terra::extract(idw_smoothed_25, gt_points)
gt_points$IDW_LiDAR_Canopy_smoothed_25 <- gt_points$Z_max_veg_25 - gt_points$idw_smoothed_25_bathy

# creating datasets for figure
LiDAR_canopy_25 <- gt_points %>% select(mnl_hgh, IDW_LiDAR_Canopy_smoothed_25) %>% mutate(dataset = "25 m")
names(LiDAR_canopy_25) <- c("manual_height", "LiDAR_height", "geometry", "dataset")
LiDAR_canopy_25_df <- data.frame(
  manual_height = LiDAR_canopy_25$manual_height,
  LiDAR_height = LiDAR_canopy_25$LiDAR_height$focal_mean,
  dataset = LiDAR_canopy_25$dataset)

LiDAR_canopy_50 <- gt_points %>% select(mnl_hgh, IDW_LiDAR_Canopy_smoothed) %>% mutate(dataset = "50 m")
names(LiDAR_canopy_50) <- c("manual_height", "LiDAR_height", "geometry", "dataset")
LiDAR_canopy_50_df <- data.frame(
  manual_height = LiDAR_canopy_50$manual_height,
  LiDAR_height = LiDAR_canopy_50$LiDAR_height$focal_mean,
  dataset = LiDAR_canopy_50$dataset)

# join
LiDAR_canopy_fig <- rbind(LiDAR_canopy_25_df, LiDAR_canopy_50_df)
LiDAR_canopy_fig$LiDAR_height <- LiDAR_canopy_fig$LiDAR_height * 100
LiDAR_canopy_fig$dataset <- as.factor(LiDAR_canopy_fig$dataset)


Figure_6A <- ggplot(LiDAR_canopy_fig, aes(x = manual_height*0.01, y = LiDAR_height*0.01, 
                                          color = dataset, shape = dataset)) +
  geom_point(size = 3) +
  geom_smooth(method = "lm", se = TRUE) +
  scale_color_manual(values = c("25 m" = "sienna2", "50 m" = "slategrey")) +
  scale_shape_manual(values = c("25 m" = 16, "50 m" = 17)) +
  labs(x = "Manual canopy height (m)",
       y = "LiDAR canopy height (m)",
       title = "A",
       color = "LiDAR flight elevation",
       shape = "LiDAR flight elevation") +
  scale_x_continuous(limits = c(-0.06, 0.60), breaks = seq(0, 0.60, by = 0.10)) +
  scale_y_continuous(limits = c(-0.06, 0.60), breaks = seq(0, 0.60, by = 0.10)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +
  theme_classic() +
  theme(
    axis.text = element_text(size = 18),
    axis.title = element_text(size = 20),
    title = element_text(size = 28),
    legend.text = element_text(size = 14),
    legend.title = element_text(size = 16),
    legend.position = c(0.3, 0.9),
    plot.margin = margin(0.1, 0.1, 0.1, 0.1, "cm"))

Figure_6A


# Calculating error between manual canopy height and LiDAR canopy height

canopy_error_summary <- LiDAR_canopy_fig %>%
  mutate(error = LiDAR_height - manual_height) %>%
  group_by(dataset) %>%
  summarise(
    mean_error = mean(error, na.rm = TRUE),
    sd_error = sd(error, na.rm = TRUE),
    rmse = sqrt(mean(error^2, na.rm = TRUE)),
    n = n()
  ) %>%
  ungroup()

print(canopy_error_summary)



# Creating Digital Surface Model of vegetation
# using lidR 'rasterize_canopy'

# p2r algorithm
# see lidR documentation pdf, page 34, for explanation of p2r algo

#################################
# Fix UTF-8 encoding issues
vegetation_las <- fix_crs_encoding(vegetation_las)
#################################

# 50 m data
dsm_p2r <- rasterize_canopy(vegetation_las, res = 0.1, algorithm = p2r(subcircle = 0.5))
plot_dtm3d(dsm_p2r, col = "green") |> add_dtm3d(idw_smoothed) 

dsm_p2r_aligned <- crop(dsm_p2r, ext(idw_smoothed))
ext(dsm_p2r_aligned)
ext(idw_smoothed)

idw_aligned <- crop(idw_smoothed, ext(dsm_p2r_aligned))
ext(dsm_p2r_aligned)
ext(idw_aligned)

dsm_masked <- mask(dsm_p2r_aligned, idw_aligned)
crs(dsm_masked)

# load crop area (to remove Fucus vesiculosus and F. serratus from modelled area)
crop_area <- st_read("crop_area.shp")
crop_area_2 <- st_read("crop_area_2.shp")
plot(crop_area)
crop <- st_transform(crop_area, crs(dsm_masked))
crop_2 <- st_transform(crop_area_2, crs(dsm_masked))

# Mask the raster to remove the area inside the polygon (inverse)
final_raster <- mask(dsm_masked, crop, inverse = TRUE)
DSM_fin <- mask(final_raster, crop_2, inverse = TRUE) 
crs(DSM_fin)
DSM_final <- project(DSM_fin, "EPSG:32632")

# ensure extents align
# Align idw_smoothed to DSM_final
idw_smoothed_aligned <- resample(idw_smoothed, DSM_final)

# Ensure the extents match
DSM_final_cropped <- crop(DSM_final, idw_smoothed_aligned)
idw_smoothed_cropped <- crop(idw_smoothed_aligned, DSM_final_cropped)
crs(DSM_final_cropped)
crs(idw_smoothed_cropped)

crs(DSM_final_cropped) <- crs(idw_smoothed_cropped)

# Summary of final DSM layer
summary(DSM_final_cropped)
plot(DSM_final_cropped)

# summary of final DTM layer
summary(idw_smoothed_cropped)
plot(idw_smoothed_cropped)


# Creating canopy height raster
canopy_height_raster <- DSM_final_cropped - idw_smoothed_cropped
# Plot the result
plot(canopy_height_raster)

# plotting sub-area
plot(canopy_height_raster, xlim = c(564950, 564970), ylim = c(6541560, 6541580))


# removing areas with canopy less than or equal to 0
# Create a new raster with values > 0
canopy_above_zero <- canopy_height_raster
canopy_above_zero[canopy_above_zero <= 0] <- NA 
plot(canopy_above_zero, main = "Canopy Height (m)")


############
# 25 m data

#################################
# Fix UTF-8 encoding issues
vegetation_las_25 <- fix_crs_encoding(vegetation_las_25)
#################################


dsm_p2r_25 <- rasterize_canopy(vegetation_las_25, res = 0.1, algorithm = p2r(subcircle = 0.5))
plot_dtm3d(dsm_p2r_25, col = "green") |> add_dtm3d(idw_smoothed_25) 

dsm_p2r_25_aligned <- crop(dsm_p2r_25, ext(idw_smoothed_25))
ext(dsm_p2r_25_aligned)
ext(idw_smoothed_25)

idw_aligned_25 <- crop(idw_smoothed_25, ext(dsm_p2r_25_aligned))
ext(dsm_p2r_25_aligned)
ext(idw_aligned_25)


dsm_masked_25 <- mask(dsm_p2r_25_aligned, idw_aligned_25)
crs(dsm_masked_25)

# load crop area
crop_area <- st_read("crop_area.shp")
crop_area_2 <- st_read("crop_area_2.shp")
plot(crop_area)
crop <- st_transform(crop_area, crs(dsm_masked_25))
crop_2 <- st_transform(crop_area_2, crs(dsm_masked_25))

# Mask the raster to remove the area inside the polygon (inverse)
final_raster_25 <- mask(dsm_masked_25, crop, inverse = TRUE)
DSM_fin_25 <- mask(final_raster_25, crop_2, inverse = TRUE) 
crs(DSM_fin_25)
DSM_final_25 <- project(DSM_fin_25, "EPSG:32632")

# ensure extents align
# Align idw_smoothed to DSM_final
idw_smoothed_aligned_25 <- resample(idw_smoothed_25, DSM_final_25)

# Ensure the extents match
DSM_final_cropped_25 <- crop(DSM_final_25, idw_smoothed_aligned_25)
idw_smoothed_cropped_25 <- crop(idw_smoothed_aligned_25, DSM_final_cropped_25)
crs(DSM_final_cropped_25)
crs(idw_smoothed_cropped_25)

crs(DSM_final_cropped_25) <- crs(idw_smoothed_cropped_25)

# Summary of final DSM layer
summary(DSM_final_cropped_25)
plot(DSM_final_cropped_25)

# summary of final DTM layer
summary(idw_smoothed_cropped_25)
plot(idw_smoothed_cropped_25)


# Creating canopy height raster
canopy_height_raster_25 <- DSM_final_cropped_25 - idw_smoothed_cropped_25
# Plot the result
plot(canopy_height_raster_25)

# plotting sub-area
plot(canopy_height_raster_25, xlim = c(564950, 564970), ylim = c(6541560, 6541580))

# removing areas with canopy less than or equal to 0
# Create a new raster with values > 0
canopy_above_zero_25 <- canopy_height_raster_25
canopy_above_zero_25[canopy_above_zero_25 <= 0] <- NA 
plot(canopy_above_zero_25, main = "Canopy Height (m)")






# 50m and 25m canopy overlap
# Use the idw smoothed cropped DTM layers
# to find smallest overlapping area beteween
# the two canopy height rasters

# First, let's check their properties
print("25m raster extent:")
print(ext(idw_smoothed_cropped_25))
print("Regular raster extent:")
print(ext(idw_smoothed_cropped))

# Create binary masks where TRUE means we have data
mask_25 <- !is.na(idw_smoothed_cropped_25)
mask_reg <- !is.na(idw_smoothed_cropped)

# Resample the 25m mask to match regular raster
mask_25_resampled <- resample(mask_25, mask_reg, method="near")

# Find true overlap - only TRUE where BOTH have data
overlap_mask <- mask_25_resampled & mask_reg
values(overlap_mask)[!values(overlap_mask)] <- NA

# First align both canopy layers to the SAME extent (using the 50m extent as reference)
canopy_25_aligned <- resample(canopy_above_zero_25, canopy_above_zero)

# Now mask both using the same overlap mask
canopy_50_masked <- mask(canopy_above_zero, overlap_mask)
canopy_25_masked <- mask(canopy_25_aligned, overlap_mask)

# Verify the extents are the same
print("50m masked extent:")
print(ext(canopy_50_masked))
print("25m masked extent:")
print(ext(canopy_25_masked))

# Plot to verify
par(mfrow=c(1,2))
plot(canopy_50_masked, main="50m masked canopy")
plot(canopy_25_masked, main="25m masked canopy")
par(mfrow=c(1,1))

print(ext(canopy_25_masked))
print(ext(canopy_50_masked))

# summary values of eelgrass raster heights
summary(values(canopy_50_masked$Z, na.rm = TRUE))
summary(values(canopy_25_masked$Z, na.rm = TRUE))


# calculating volume and area of 50m and 25m eelgrass canopies

# resolution is the same for the 50m and 25m data
res_x <- res(canopy_25_masked)[1]  # Cell width
res_y <- res(canopy_25_masked)[2]  # Cell height

# Calculate area per cell (mB2)
cell_area <- res_x * res_y

# number of raster cells
non_na_cells_50 <- sum(!is.na(values(canopy_50_masked)))
non_na_cells_25 <- sum(!is.na(values(canopy_25_masked)))

# canopy heights
height_values_50 <- values(canopy_50_masked)
height_values_50 <- height_values_50[!is.na(height_values_50)]
height_values_50 <- pmax(height_values_50, 0)
height_values_25 <- values(canopy_25_masked)
height_values_25 <- height_values_25[!is.na(height_values_25)]
height_values_25 <- pmax(height_values_25, 0)



# Total area
total_area_50 <- non_na_cells_50 * cell_area
cat("Total Area (mB2) 50m data:", formatC(total_area_50, format="f", digits=2), "\n")
total_area_25 <- non_na_cells_25 * cell_area
cat("Total Area (mB2) 25m data:", formatC(total_area_25, format="f", digits=2), "\n")

# Total volume
total_volume_50 <- sum(height_values_50 * cell_area)
cat("Total Volume (mB3) 50m data:", formatC(total_volume_50, format="f", digits=2), "\n")
total_volume_25 <- sum(height_values_25 * cell_area)
cat("Total Volume (mB3) 25m data:", formatC(total_volume_25, format="f", digits=2), "\n")


# Creating total area and volume figure
area_50 <- c(3881,4354,4520,4752,4909)
area_25 <- c(4722,5130,5341,5536,5703)

volume_50 <- c(628,728,771,822,861)
volume_25 <- c(818,931,993,1049,1099)

subcircle <- c(0.1, 0.2, 0.3, 0.4, 0.5)

df_50m <- data.frame(
  Area = area_50,
  Volume = volume_50,
  Data = "50 m",
  Subcircle = subcircle
)

df_25m <- data.frame(
  Area = area_25,
  Volume = volume_25,
  Data = "25 m",
  Subcircle = subcircle
)

area_volume <- rbind(df_50m, df_25m)

# Sort by Subcircle value and Data
area_volume <- area_volume[order(area_volume$Subcircle, area_volume$Data), ]

# View the result
print(area_volume)

scale_factor <- max(area_volume$Volume) / max(area_volume$Area)

ggplot(area_volume, aes(x = Subcircle * 100)) +
  # Volume lines and points
  geom_point(aes(y = Volume, color = Data, shape = Data), size = 3) +
  geom_line(aes(y = Volume, color = Data, group = Data), linewidth = 1) +
  # Area lines and points (scaled)
  geom_point(aes(y = Area * scale_factor, shape = Data), size = 3) +
  geom_line(aes(y = Area * scale_factor, group = Data), linetype = "dashed", linewidth = 1) +
  scale_color_manual(values = c("25 m" = "sienna2", "50 m" = "slategrey")) +
  scale_shape_manual(values = c("25 m" = 16, "50 m" = 17)) +
  # Primary y-axis (Volume)
  scale_y_continuous(
    name = bquote("Eelgrass volume (m"^3*")"),
    # Secondary y-axis (Area)
    sec.axis = sec_axis(~./scale_factor, 
                        name = bquote("Eelgrass area (m"^2*")"),
                        breaks = seq(0, 6000, by = 400))) +
  labs(
    x = "DSM smoothing radius (cm)",,
    title = "B",
    color = "LiDAR flight elevation",
    shape = "LiDAR flight elevation"
  ) +
  theme_classic() +
  theme(
    axis.text = element_text(size = 18),
    axis.text.y.right = element_text(size = 18),
    title = element_text(size = 28),
    axis.title = element_text(size = 20),
    axis.title.y.right = element_text(size = 20),
    legend.position = "none",
    plot.margin = margin(0.1, 0.1, 0.1, 0.1, "cm"))



# Creating canopy height value histograms for both rasters
max(c(
  hist(height_values_25, breaks = 30, plot = FALSE)$counts,
  hist(height_values_50, breaks = 30, plot = FALSE)$counts))


# 25m
Figure_6C <- ggplot(data.frame(height = height_values_25), aes(x = height)) +
  geom_histogram(fill = "sienna2", bins = 40) +
  labs(x = "Canopy height (m)",
       y = "",
       title = "C" ) +
  scale_x_continuous(breaks = seq(0, 1, by = 0.1), limits = c(0, 1.2)) +
  scale_y_continuous(breaks = c(0, 20000, 40000, 60000, 80000), 
                     limits = c(0, 90000)) +
  theme_classic() +
  theme(
    axis.text = element_text(size = 18),
    axis.title = element_text(size = 20),
    title = element_text(size = 28),
    plot.margin = margin(0.1, 0.1, 0.1, 0.1, "cm")) +
  annotate("text", x = 0.4, y = 80000, label = "25 m flight, \n0.5 m DSM smoothing radius", size = 6)

Figure_6C

# 50m
Figure_6D <- ggplot(data.frame(height = height_values_50), aes(x = height)) +
  geom_histogram(fill = "slategrey", bins = 40) +
  labs(x = "Canopy height (m)",
       y = "",
       title = "D" ) +
  scale_x_continuous(breaks = seq(0, 1, by = 0.1), limits = c(0, 1.2)) +
  scale_y_continuous(breaks = c(0, 20000, 40000, 60000, 80000), 
                     limits = c(0, 90000)) +
  theme_classic() +
  theme(
    axis.text = element_text(size = 18),
    axis.title = element_text(size = 20),
    title = element_text(size = 28),
    plot.margin = margin(0.1, 0.1, 0.1, 0.1, "cm")) +
  annotate("text", x = 0.4, y = 80000, label = "50 m flight, \n0.5 m DSM smoothing radius", size = 6)


Figure_6D


# Aligning IDW DTM rasters
idw_smoothed_cropped_aligned <- resample(idw_smoothed_cropped, overlap_mask)
idw_smoothed_cropped_25_aligned <- resample(idw_smoothed_cropped_25, overlap_mask)

# masking
idw_final <- mask(idw_smoothed_cropped_aligned, overlap_mask)
idw_final_25 <- mask(idw_smoothed_cropped_25_aligned, overlap_mask)


# Creating canopy raster and histogram figure for poster
# Convert SpatRaster to data frame
canopy_50_df <- as.data.frame(canopy_50_masked, xy = TRUE)
colnames(canopy_50_df)[3] <- "value"

dtm_50_df <- as.data.frame(idw_final, xy = TRUE)
colnames(dtm_50_df)[3] <- "value"

# Plot with ggplot2
canopy_raster_poster <- ggplot() +
  geom_raster(data = dtm_50_df, aes(x = x, y = y, fill = value*100), alpha = 0.5) +
  scale_fill_gradientn(
    colors = gray.colors(100),
    name = "Bathymetry\n(cm)") +
  new_scale_fill() +
  geom_raster(data = canopy_50_df, aes(x = x, y = y, fill = value*100)) +
  scale_fill_gradientn(
    colors = c("#00441b", "#a1d76a", "#ffffbf", "#ffd700"),
    name = "Cell resolution = 10 cm\nCanopy height (cm)") +
  coord_equal() +
  labs(x = "Eastings", y = "Northings") +
  theme_classic() +
  theme(
    panel.background = element_rect(fill = "transparent", color = NA),
    plot.background = element_rect(fill = "transparent", color = NA),
    axis.title = element_text(size = 6),
    legend.position = c(0.9, 0.95),
    legend.justification = c("right", "top"),
    legend.background = element_rect(fill = "transparent", color = NA),
    legend.key = element_rect(fill = "transparent", color = NA),
    legend.key.height = unit(0.6, "cm"),
    legend.key.width = unit(0.5, "cm"),
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 9),
    legend.margin = margin(0, 0, 0, 0),
    legend.box.margin = margin(0, 0, 0, 0),
    legend.box = "horizontal")

canopy_hist_poster <- ggplot(data.frame(height = height_values_50), aes(x = height*100)) +
  geom_histogram(fill = "slategrey", bins = 40) +
  labs(x = "Canopy height (cm)",
       y = "Raster cell frequency") +
  scale_x_continuous(breaks = seq(0, 60, by = 10), limits = c(0, 60)) +
  scale_y_continuous(breaks = c(0, 10000, 20000, 30000), 
                     limits = c(0, 30000)) +
  theme_classic() +
  theme(
    axis.text = element_text(size = 8),
    axis.title = element_text(size = 10),
    title = element_text(size = 28),
    plot.margin = margin(0.1, 0.1, 0.1, 0.1, "cm"),
    panel.background = element_rect(fill = "transparent", color = NA),
    plot.background = element_rect(fill = "transparent", color = NA))


hist_grob <- ggplotGrob(canopy_hist_poster)

poster_figure <- canopy_raster_poster +
  annotation_custom(
    grob = hist_grob,
    xmin = min(canopy_50_df$x) - 0.05 * diff(range(canopy_50_df$x)),
    xmax = min(canopy_50_df$x) + 0.32 * diff(range(canopy_50_df$x)),
    ymin = min(canopy_50_df$y) - 0.05 * diff(range(canopy_50_df$y)),
    ymax = min(canopy_50_df$y) + 0.32 * diff(range(canopy_50_df$y)))


#ggsave("canopy_raster_poster.png", plot = poster_figure, width = 14, height = 8, dpi = 300, bg = "transparent")





# Plot Figure 4, canopy height above IDW bottom

# For 50m data
canopy_50_df <- as.data.frame(canopy_50_masked*100, xy = TRUE) %>%
  rename(canopy_height = 3) %>% 
  mutate(resolution = "50m")

# For 25m data
canopy_25_df <- as.data.frame(canopy_25_masked*100, xy = TRUE) %>%
  rename(canopy_height = 3) %>%
  mutate(resolution = "25m")

# Bathy from 50m
bath_50_df <- as.data.frame(idw_final*100, xy = TRUE) %>%
  rename(bathymetry = 3) %>%
  mutate(resolution = "50m")

# Bathy from 25m
bath_25_df <- as.data.frame(idw_final_25*100, xy = TRUE) %>%
  rename(bathymetry = 3) %>%
  mutate(resolution = "25m")


Figure_4_final <- ggplot() +
  # Add bathymetry layer FIRST (so it goes on bottom)
  geom_raster(data = rbind(bath_50_df, bath_25_df), 
              aes(x = x, y = y, fill = bathymetry*0.01)) +
  scale_fill_gradientn(colors = gray.colors(50, start = 0, end = 1, alpha = 0.6),
                       name = "Bathymetry\n(m)") +
  # Add canopy layer SECOND (so it goes on top)
  new_scale_fill() +
  geom_raster(data = rbind(canopy_50_df, canopy_25_df), 
              aes(x = x, y = y, fill = canopy_height*0.01)) +
  scale_fill_gradientn(colors = viridis(100, alpha = 0.9),
                       limits = c(0, 1.00),
                       name = "Canopy\nHeight (m)") +
  facet_wrap(~resolution, ncol = 2) +
  coord_equal() +
  theme_classic() +
  labs(x = "Eastings",
       y = "Northings",
       subtitle = "Canopy smoothing radius = 0.5 m") +
  theme(legend.position = "right",
        strip.text = element_text(size = 12, face = "bold"),
        axis.text = element_text(size = 10),
        axis.title = element_text(size = 12),
        plot.subtitle = element_text(size = 16, hjust = 0.5),
        legend.title = element_text(size = 14),
        legend.text = element_text(size = 12))


#ggsave("Figure_4.png", plot = Figure_4_final, device = "png", width = 15, height = 8, dpi = 300)


# Figure with darker bathymetry
#Figure_4_final <- ggplot() +
#  # Add bathymetry layer FIRST (so it goes on bottom)
#  geom_raster(data = rbind(bath_50_df, bath_25_df), 
#              aes(x = x, y = y, fill = bathymetry*0.01)) +
#  scale_fill_gradientn(colors = scales::alpha(c("black", "gray15", "gray30", "gray45", "gray60", "gray75", "gray90", "white"), 
#                                               alpha = 0.75),
#                       name = "Bathymetry\n(m)") +
#  # Add canopy layer SECOND (so it goes on top)
#  new_scale_fill() +
#  geom_raster(data = rbind(canopy_50_df, canopy_25_df), 
#              aes(x = x, y = y, fill = canopy_height*0.01)) +
#  scale_fill_gradientn(colors = viridis(100, alpha = 0.9),
#                       limits = c(0, 1.00),
#                       name = "Canopy\nHeight (m)") +
#  facet_wrap(~resolution, ncol = 2) +
#  coord_equal() +
#  theme_classic() +
#  labs(x = "Eastings",
#       y = "Northings",
#       subtitle = "Canopy smoothing radius = 0.5 m") +
#  theme(legend.position = "right",
#        strip.text = element_text(size = 12, face = "bold"),
#        axis.text = element_text(size = 10),
#        axis.title = element_text(size = 12),
#        plot.subtitle = element_text(size = 16, hjust = 0.5),
#        legend.title = element_text(size = 14),
#        legend.text = element_text(size = 12))

# Display the plot
#windows()
#print(Figure_4_final)
#dev.off()

#ggsave("Figure_4_new_darker.png", plot = Figure_4_final, device = "png", width = 15, height = 8, dpi = 300)

#' Process LiDAR data to calculate canopy stats and return key outputs.
#'
#' @param subcircle_val The subcircle radius for the p2r algorithm.
#' @param vegetation_las The 50m vegetation LAS object.
#' @param idw_smoothed The 50m smoothed DTM raster.
#' @param vegetation_las_25 The 25m vegetation LAS object.
#' @param idw_smoothed_25 The 25m smoothed DTM raster.
#' @param crop_area The first cropping polygon.
#' @param crop_area_2 The second cropping polygon.
#' @return A list containing 'summary', 'heights', 'canopy_50m', 'canopy_25m', 'dtm_50m_cropped', and 'dtm_25m_cropped'.
process_and_summarize_canopy <- function(subcircle_val,
                                         vegetation_las, idw_smoothed,
                                         vegetation_las_25, idw_smoothed_25,
                                         crop_area, crop_area_2) {
  
  # --- 50 m data processing ---
  cat("  Processing 50m data...\n")
  dsm_p2r <- rasterize_canopy(vegetation_las, res = 0.1, algorithm = p2r(subcircle = subcircle_val))
  dsm_p2r_aligned <- crop(dsm_p2r, ext(idw_smoothed))
  idw_aligned <- crop(idw_smoothed, ext(dsm_p2r_aligned))
  dsm_masked <- mask(dsm_p2r_aligned, idw_aligned)
  
  crop <- st_transform(crop_area, crs(dsm_masked))
  crop_2 <- st_transform(crop_area_2, crs(dsm_masked))
  
  final_raster <- mask(dsm_masked, crop, inverse = TRUE)
  DSM_fin <- mask(final_raster, crop_2, inverse = TRUE)
  DSM_final <- project(DSM_fin, "EPSG:32632")
  
  idw_smoothed_aligned <- resample(idw_smoothed, DSM_final)
  DSM_final_cropped <- crop(DSM_final, idw_smoothed_aligned)
  idw_smoothed_cropped <- crop(idw_smoothed_aligned, DSM_final_cropped)
  crs(DSM_final_cropped) <- crs(idw_smoothed_cropped)
  
  canopy_height_raster <- DSM_final_cropped - idw_smoothed_cropped
  canopy_above_zero <- canopy_height_raster
  canopy_above_zero[canopy_above_zero <= 0] <- NA
  
  # --- 25 m data processing ---
  cat("  Processing 25m data...\n")
  dsm_p2r_25 <- rasterize_canopy(vegetation_las_25, res = 0.1, algorithm = p2r(subcircle = subcircle_val))
  dsm_p2r_25_aligned <- crop(dsm_p2r_25, ext(idw_smoothed_25))
  idw_aligned_25 <- crop(idw_smoothed_25, ext(dsm_p2r_25_aligned))
  dsm_masked_25 <- mask(dsm_p2r_25_aligned, idw_aligned_25)
  
  crop_25 <- st_transform(crop_area, crs(dsm_masked_25))
  crop_2_25 <- st_transform(crop_area_2, crs(dsm_masked_25))
  
  final_raster_25 <- mask(dsm_masked_25, crop_25, inverse = TRUE)
  DSM_fin_25 <- mask(final_raster_25, crop_2_25, inverse = TRUE)
  DSM_final_25 <- project(DSM_fin_25, "EPSG:32632")
  
  idw_smoothed_aligned_25 <- resample(idw_smoothed_25, DSM_final_25)
  DSM_final_cropped_25 <- crop(DSM_final_25, idw_smoothed_aligned_25)
  idw_smoothed_cropped_25 <- crop(idw_smoothed_aligned_25, DSM_final_cropped_25)
  crs(DSM_final_cropped_25) <- crs(idw_smoothed_cropped_25)
  
  canopy_height_raster_25 <- DSM_final_cropped_25 - idw_smoothed_cropped_25
  canopy_above_zero_25 <- canopy_height_raster_25
  canopy_above_zero_25[canopy_above_zero_25 <= 0] <- NA
  
  # --- Overlap processing ---
  cat("  Calculating overlap...\n")
  mask_25 <- !is.na(idw_smoothed_cropped_25)
  mask_reg <- !is.na(idw_smoothed_cropped)
  mask_25_resampled <- resample(mask_25, mask_reg, method="near")
  overlap_mask <- mask_25_resampled & mask_reg
  values(overlap_mask)[!values(overlap_mask)] <- NA
  
  canopy_25_aligned <- resample(canopy_above_zero_25, canopy_above_zero)
  canopy_50_masked <- mask(canopy_above_zero, overlap_mask)
  canopy_25_masked <- mask(canopy_25_aligned, overlap_mask)
  
  # --- Volume and Area Calculation ---
  cat("  Calculating area and volume...\n")
  res_x <- res(canopy_25_masked)[1]
  res_y <- res(canopy_25_masked)[2]
  cell_area <- res_x * res_y
  
  height_values_50 <- values(canopy_50_masked)
  height_values_50 <- height_values_50[!is.na(height_values_50)]
  height_values_50 <- pmax(height_values_50, 0)
  height_values_25 <- values(canopy_25_masked)
  height_values_25 <- height_values_25[!is.na(height_values_25)]
  height_values_25 <- pmax(height_values_25, 0)
  
  non_na_cells_50 <- length(height_values_50)
  non_na_cells_25 <- length(height_values_25)
  
  total_area_50 <- non_na_cells_50 * cell_area
  total_area_25 <- non_na_cells_25 * cell_area
  total_volume_50 <- sum(height_values_50 * cell_area)
  total_volume_25 <- sum(height_values_25 * cell_area)
  
  # --- Create Data Frames for output ---
  
  # Summary data frame (long format)
  summary_df <- data.frame(
    subcircle = c(subcircle_val, subcircle_val),
    Dataset = c("50 m", "25 m"),
    Area = c(total_area_50, total_area_25),
    Volume = c(total_volume_50, total_volume_25)
  )
  
  # Heights data frame (long format)
  heights_50_df <- data.frame(
    subcircle = subcircle_val,
    Dataset = "50 m",
    height = height_values_50
  )
  heights_25_df <- data.frame(
    subcircle = subcircle_val,
    Dataset = "25 m",
    height = height_values_25
  )
  heights_df <- rbind(heights_50_df, heights_25_df)
  
  return(list(
    summary = summary_df, 
    heights = heights_df,
    canopy_50m = canopy_above_zero,
    canopy_25m = canopy_above_zero_25,
    dtm_50m_cropped = idw_smoothed_cropped,
    dtm_25m_cropped = idw_smoothed_cropped_25
  ))
}


# Load crop areas once
#crop_area <- st_read("crop_area.shp")
#crop_area_2 <- st_read("crop_area_2.shp")

# Define the subcircle values to iterate over
subcircle_values <- c(0.1, 0.2, 0.3, 0.4, 0.5)

# Initialize lists to store results from each iteration
summary_list <- list()
heights_list <- list()
rasters_list <- list()
dtms_list <- list()

# Loop through each subcircle value and process the data
for (sc_val in subcircle_values) {
  cat("\nProcessing with subcircle value:", sc_val, "\n")
  
  # Process data and get all results
  iteration_results <- process_and_summarize_canopy(
    subcircle_val = sc_val,
    vegetation_las = vegetation_las,
    idw_smoothed = idw_smoothed,
    vegetation_las_25 = vegetation_las_25,
    idw_smoothed_25 = idw_smoothed_25,
    crop_area = crop_area,
    crop_area_2 = crop_area_2
  )
  
  # Store results
  summary_list[[as.character(sc_val)]] <- iteration_results$summary
  heights_list[[as.character(sc_val)]] <- iteration_results$heights
  rasters_list[[as.character(sc_val)]] <- list(
    raster_50m = iteration_results$canopy_50m,
    raster_25m = iteration_results$canopy_25m
  )
  # Store the DTMs from each iteration
  dtms_list[[as.character(sc_val)]] <- list(
    dtm_50m = iteration_results$dtm_50m_cropped,
    dtm_25m = iteration_results$dtm_25m_cropped
  )
}

# Combine the lists of data frames into single data frames
summary_df <- do.call(rbind, summary_list)
all_heights_df <- do.call(rbind, heights_list)
rownames(summary_df) <- NULL
rownames(all_heights_df) <- NULL

# Print the summary table
cat("\n--- Summary of Canopy Area and Volume ---\n")
print(summary_df)

scale_factor <- max(summary_df$Volume) / max(summary_df$Area)

# Figure 6B, volume and area plot
Figure_6B <- ggplot(summary_df, aes(x = subcircle)) +
  # Volume lines and points
  geom_point(aes(y = Volume, color = Dataset, shape = Dataset), size = 3) +
  geom_line(aes(y = Volume, color = Dataset, group = Dataset), linewidth = 1) +
  # Area lines and points (scaled)
  geom_point(aes(y = Area * scale_factor, shape = Dataset), size = 3) +
  geom_line(aes(y = Area * scale_factor, group = Dataset), linetype = "dashed", linewidth = 1) +
  scale_color_manual(values = c("25 m" = "sienna2", "50 m" = "slategrey")) +
  scale_shape_manual(values = c("25 m" = 16, "50 m" = 17)) +
  # Primary y-axis (Volume)
  scale_y_continuous(
    name = bquote("Eelgrass volume (m"^3*")"),
    # Secondary y-axis (Area)
    sec.axis = sec_axis(~./scale_factor, 
                        name = bquote("Eelgrass area (m"^2*")"),
                        breaks = seq(0, 6000, by = 400))) +
  labs(
    x = "DSM smoothing radius (m)",,
    title = "B",
    color = "LiDAR flight elevation",
    shape = "LiDAR flight elevation") +
  theme_classic() +
  theme(
    axis.text = element_text(size = 18),
    axis.text.y.right = element_text(size = 18),
    title = element_text(size = 28),
    axis.title = element_text(size = 20),
    axis.title.y.right = element_text(size = 20),
    legend.position = "none",
    plot.margin = margin(0.1, 0.1, 0.1, 0.1, "cm"))


Figure_6B


# saving Figure 6
Figure_6 <- ggarrange(Figure_6A, Figure_6B,
                      Figure_6C, Figure_6D,
                      nrow=2, ncol=2, align = "hv")

Figure_6

#ggsave("Figure_6.png", plot = Figure_6, device = "png", width = 16, height = 10, dpi = 300)


# summary of area and volume for each dataset
summary_df %>% group_by(Dataset) %>% summarise(mean_area = mean(Area), mean_area_sd = sd(Area),
mean_volume = mean(Volume), mean_volume_sd = sd(Volume))


# Comparing Sept 2023 and Nov 2024 seagrass canopy height
canopy_sept <- read.csv("ZOSMA_height_Sept_2023.csv")

# Extract coordinates from both datasets
canopy_coords <- canopy_sept[, c("X", "Y")]  # Coordinates from canopy_sept
gt_coords <- st_coordinates(gt_points)  # Extract coordinates from gt_points (simple features)

# Perform nearest neighbor search
nn_result <- nn2(data = gt_coords, query = canopy_coords, k = 1)

# Extract indices of nearest points
nearest_indices <- nn_result$nn.idx

# Add nearest X, Y, distance, and Z to canopy_sept
canopy_sept$nearest_X <- gt_coords[nearest_indices, 1]  # X coordinate of nearest point
canopy_sept$nearest_Y <- gt_coords[nearest_indices, 2]  # Y coordinate of nearest point
canopy_sept$distance_to_nearest <- nn_result$nn.dists  # Distance to nearest point
canopy_sept$nearest_Z <- gt_points$Z_min_veg[nearest_indices]  # Z value from gt_points
canopy_sept$nearest_mnl_hgh <- gt_points$mnl_hgh[nearest_indices]


# View the updated canopy_sept dataframe
head(canopy_sept)

# If you want to filter based on distance (e.g., distance < 0.1)
canopy_overlap <- canopy_sept %>%
  dplyr::filter(distance_to_nearest < 0.75)

# View the subset of points with nearest neighbors within 0.1 distance
head(canopy_overlap)

S_Figure_5 <- ggplot(data=canopy_overlap, aes(x=Height_max*0.01, y=nearest_mnl_hgh*0.01)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE) + 
  labs(x = "September 2023 canopy height (m)",
       y = "November 2024 canopy height (m)") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +
  scale_x_continuous(limits = c(0, 0.62), breaks = seq(0, 0.60, by = 0.10)) +
  scale_y_continuous(limits = c(0, 0.62), breaks = seq(0, 0.60, by = 0.10)) +
  theme_classic() +
  theme(
    axis.text = element_text(size = 18),
    axis.title = element_text(size = 20))

print(S_Figure_5)
  
  #ggsave("S_Figure_5.png", plot = S_Figure_5, device = "png", width = 8, height = 8, dpi = 300)

# Calculate R-squared value for the linear relationship in S_Figure_5
cat("\n=== R-squared value for S_Figure_5 (September 2023 vs November 2024 canopy height) ===\n")
lm_S5 <- lm(nearest_mnl_hgh ~ Height_max, data = canopy_overlap)
r_squared_S5 <- summary(lm_S5)$r.squared
adj_r_squared_S5 <- summary(lm_S5)$adj.r.squared
slope_S5 <- coef(lm_S5)[2]
intercept_S5 <- coef(lm_S5)[1]

cat(sprintf("R² = %.4f\n", r_squared_S5))
cat(sprintf("Adjusted R² = %.4f\n", adj_r_squared_S5))
cat(sprintf("Slope = %.4f\n", slope_S5))
cat(sprintf("Intercept = %.4f\n", intercept_S5))
cat(sprintf("Model: November 2024 = %.4f × September 2023 + %.4f\n", slope_S5, intercept_S5))
cat("\n")


# Calculating vertical errors between canopy overlap
error_summary <- canopy_overlap %>%
  mutate(error = (Height_max - nearest_mnl_hgh)) %>%  # error in cm
  summarise(
    mean_error = mean(error, na.rm = TRUE),
    sd_error = sd(error, na.rm = TRUE),
    rmse = sqrt(mean(error^2, na.rm = TRUE)),
    n = n()
  ) %>%
  ungroup()

print(error_summary)



# loading eelgrass volumetric biomass values from 2023 field work
biomass_values <- read.csv("Eelgrass_biomass_2023_KH_HG.csv")
summary(biomass_values)

# Get numeric columns only
numeric_cols <- sapply(biomass_values, is.numeric)
numeric_data <- biomass_values[, numeric_cols]

# Calculate statistics for each numeric variable
stats_summary <- data.frame(
  Variable = character(),
  N = numeric(),
  Mean = numeric(),
  SD = numeric(),
  CI_Lower = numeric(),
  CI_Upper = numeric(),
  stringsAsFactors = FALSE
)

for(col_name in names(numeric_data)) {
  x <- numeric_data[[col_name]]
  x_clean <- x[!is.na(x)]
  
  if(length(x_clean) > 0) {
    n <- length(x_clean)
    mean_val <- mean(x_clean)
    sd_val <- sd(x_clean)
    se_val <- sd_val / sqrt(n)
    t_val <- qt(0.975, df = n - 1)
    
    ci_lower <- mean_val - t_val * se_val
    ci_upper <- mean_val + t_val * se_val
    
    stats_summary <- rbind(stats_summary, data.frame(
      Variable = col_name,
      N = n,
      Mean = round(mean_val, 3),
      SD = round(sd_val, 3),
      CI_Lower = round(ci_lower, 3),
      CI_Upper = round(ci_upper, 3)
    ))
  }
}

print(stats_summary, row.names = FALSE)

mean_Bvol <- mean(biomass_values$Biomass.density..g.ww.m.3.)
min_Bvol <- min(biomass_values$Biomass.density..g.ww.m.3.)
max_Bvol <- max(biomass_values$Biomass.density..g.ww.m.3.)


# Multiplying canopy volume values to Bvol values for
# biomass estimation

# Summarising for subcircle = 0.5 only
subcircle_05_data <- summary_df[summary_df$subcircle == 0.5, ]

# Create the results dataframe
biomass_results <- data.frame(
  subcircle = subcircle_05_data$subcircle,
  Dataset = subcircle_05_data$Dataset,
  Area = subcircle_05_data$Area,
  Volume = subcircle_05_data$Volume,
  mean_Bvol_ww = subcircle_05_data$Volume * mean_Bvol*0.001,
  min_Bvol_ww = subcircle_05_data$Volume * min_Bvol*0.001,
  max_Bvol_ww = subcircle_05_data$Volume * max_Bvol*0.001)

# Display the results
print(biomass_results)

# Calculating results
biomass_results <- biomass_results %>% 
  mutate(mean_Bvol_ww_per_m = mean_Bvol_ww/Area,
         min_Bvol_ww_per_m = min_Bvol_ww/Area,
         max_Bvol_ww_per_m = max_Bvol_ww/Area) %>%
  mutate(mean_Bvol_dw = mean_Bvol_ww * 0.24,
         min_Bvol_dw = min_Bvol_ww * 0.24,
         max_Bvol_dw = max_Bvol_ww * 0.24) %>%
  mutate(mean_Bvol_dw_per_m = mean_Bvol_dw/Area,
         min_Bvol_dw_per_m = min_Bvol_dw/Area,
         max_Bvol_dw_per_m = max_Bvol_dw/Area) %>%
  mutate(mean_Bvol_C = mean_Bvol_dw * 0.34,
         min_Bvol_C = min_Bvol_dw * 0.34,
         max_Bvol_C = max_Bvol_dw * 0.34) %>%
  mutate(mean_Bvol_C_per_m = mean_Bvol_C/Area,
         min_Bvol_C_per_m = min_Bvol_C/Area,
         max_Bvol_C_per_m = max_Bvol_C/Area)

options(scipen = 999)   

biomass_results_table <- biomass_results %>%
  select(-subcircle, -Area, -Volume) %>%
  pivot_longer(cols = -Dataset, names_to = "variable", values_to = "value") %>%
  separate(variable, into = c("stat", "type"), sep = "_", extra = "merge") %>%
  pivot_wider(names_from = stat, values_from = value) %>%
  mutate(
    Metric = case_when(
      type == "Bvol_ww" ~ "Biomass kg ww (study area)",
      type == "Bvol_ww_per_m" ~ "Biomass kg ww m-2",
      type == "Bvol_dw" ~ "Biomass kg dw (study area)",
      type == "Bvol_dw_per_m" ~ "Biomass kg dw m-2",
      type == "Bvol_C" ~ "C kg (study area)",
      type == "Bvol_C_per_m" ~ "C g m-2"
    )
  ) %>%
  select(Dataset, Metric, Mean_Bvol = mean, Minimum_Bvol = min, Maximum_Bvol = max) %>%
  arrange(Dataset)


### WITH CI VALUES
# Get numeric columns only
numeric_cols <- sapply(biomass_values, is.numeric)
numeric_data <- biomass_values[, numeric_cols]

# Calculate statistics for each numeric variable
stats_summary <- data.frame(
  Variable = character(),
  N = numeric(),
  Mean = numeric(),
  SD = numeric(),
  CI_Lower = numeric(),
  CI_Upper = numeric(),
  stringsAsFactors = FALSE
)

for(col_name in names(numeric_data)) {
  x <- numeric_data[[col_name]]
  x_clean <- x[!is.na(x)]
  
  if(length(x_clean) > 0) {
    n <- length(x_clean)
    mean_val <- mean(x_clean)
    sd_val <- sd(x_clean)
    se_val <- sd_val / sqrt(n)
    t_val <- qt(0.975, df = n - 1)
    
    ci_lower <- mean_val - t_val * se_val
    ci_upper <- mean_val + t_val * se_val
    
    stats_summary <- rbind(stats_summary, data.frame(
      Variable = col_name,
      N = n,
      Mean = round(mean_val, 3),
      SD = round(sd_val, 3),
      CI_Lower = round(ci_lower, 3),
      CI_Upper = round(ci_upper, 3)
    ))
  }
}

print(stats_summary, row.names = FALSE)

# Extract mean and CI values for biomass density
mean_Bvol <- mean(biomass_values$Biomass.density..g.ww.m.3.)
min_Bvol <- min(biomass_values$Biomass.density..g.ww.m.3.)
max_Bvol <- max(biomass_values$Biomass.density..g.ww.m.3.)

# Get CI values for biomass density from stats_summary
bvol_stats <- stats_summary[stats_summary$Variable == "Biomass.density..g.ww.m.3.", ]
ci_lower_Bvol <- bvol_stats$CI_Lower
ci_upper_Bvol <- bvol_stats$CI_Upper

# Multiplying canopy volume values to Bvol values for biomass estimation
# Summarising for subcircle = 0.5 only
subcircle_05_data <- summary_df[summary_df$subcircle == 0.5, ]

# Create the results dataframe with CI calculations
biomass_results <- data.frame(
  subcircle = subcircle_05_data$subcircle,
  Dataset = subcircle_05_data$Dataset,
  Area = subcircle_05_data$Area,
  Volume = subcircle_05_data$Volume,
  mean_Bvol_ww = subcircle_05_data$Volume * mean_Bvol * 0.001,
  min_Bvol_ww = subcircle_05_data$Volume * min_Bvol * 0.001,
  max_Bvol_ww = subcircle_05_data$Volume * max_Bvol * 0.001,
  ci_lower_Bvol_ww = subcircle_05_data$Volume * ci_lower_Bvol * 0.001,
  ci_upper_Bvol_ww = subcircle_05_data$Volume * ci_upper_Bvol * 0.001
)

# Display the results
print(biomass_results)

# Calculating results with CI propagation
biomass_results <- biomass_results %>% 
  mutate(
    # Per m2 calculations
    mean_Bvol_ww_per_m = mean_Bvol_ww/Area,
    min_Bvol_ww_per_m = min_Bvol_ww/Area,
    max_Bvol_ww_per_m = max_Bvol_ww/Area,
    ci_lower_Bvol_ww_per_m = ci_lower_Bvol_ww/Area,
    ci_upper_Bvol_ww_per_m = ci_upper_Bvol_ww/Area
  ) %>%
  mutate(
    # Dry weight calculations
    mean_Bvol_dw = mean_Bvol_ww * 0.24,
    min_Bvol_dw = min_Bvol_ww * 0.24,
    max_Bvol_dw = max_Bvol_ww * 0.24,
    ci_lower_Bvol_dw = ci_lower_Bvol_ww * 0.24,
    ci_upper_Bvol_dw = ci_upper_Bvol_ww * 0.24
  ) %>%
  mutate(
    # Dry weight per m2 calculations
    mean_Bvol_dw_per_m = mean_Bvol_dw/Area,
    min_Bvol_dw_per_m = min_Bvol_dw/Area,
    max_Bvol_dw_per_m = max_Bvol_dw/Area,
    ci_lower_Bvol_dw_per_m = ci_lower_Bvol_dw/Area,
    ci_upper_Bvol_dw_per_m = ci_upper_Bvol_dw/Area
  ) %>%
  mutate(
    # Carbon calculations
    mean_Bvol_C = mean_Bvol_dw * 0.34,
    min_Bvol_C = min_Bvol_dw * 0.34,
    max_Bvol_C = max_Bvol_dw * 0.34,
    ci_lower_Bvol_C = ci_lower_Bvol_dw * 0.34,
    ci_upper_Bvol_C = ci_upper_Bvol_dw * 0.34
  ) %>%
  mutate(
    # Carbon per m2 calculations
    mean_Bvol_C_per_m = mean_Bvol_C/Area,
    min_Bvol_C_per_m = min_Bvol_C/Area,
    max_Bvol_C_per_m = max_Bvol_C/Area,
    ci_lower_Bvol_C_per_m = ci_lower_Bvol_C/Area,
    ci_upper_Bvol_C_per_m = ci_upper_Bvol_C/Area
  )










# Creating histogram plots from all subcircle values
cat("\n--- Creating Histogram Plot ---\n")

S_Figure_6 <- ggplot(all_heights_df, aes(x = height, fill = Dataset)) +
  geom_histogram(position = "identity", alpha = 0.6, bins = 40) +
  facet_wrap(~ subcircle, labeller = label_bquote(rows = "Smoothing radius: " ~ .(subcircle) ~ "m")) +
  labs(
    x = "Canopy height (m)",
    y = "Frequency"
  ) +
  scale_x_continuous(breaks = seq(0, 1.2, by = 0.2)) +
  scale_y_continuous() +
  scale_fill_manual(values = c("50 m" = "slategrey", "25 m" = "sienna2")) +
  coord_cartesian(xlim = c(0, 1.2)) +
  theme_classic() +
  theme(
    axis.text = element_text(size = 12),
    axis.title = element_text(size = 14),
    plot.title = element_text(size = 16, hjust = 0.5),
    legend.position = "top",
    strip.text = element_text(size = 12)
  )

print(S_Figure_6)
#ggsave("S_Figure_6.png", plot = S_Figure_6, device = "png", width = 10, height = 6, dpi = 300)

# --- Create Faceted Difference Plot (Canopy and DTM) ---
cat("\n--- Creating Canopy and DTM Difference Plot ---\n")

# Prepare data for the canopy difference plots
canopy_diff_df_list <- list()
for (sc_val in subcircle_values) {
  raster_50m <- rasters_list[[as.character(sc_val)]]$raster_50m
  raster_25m <- rasters_list[[as.character(sc_val)]]$raster_25m
  raster_25m_resampled <- resample(raster_25m, raster_50m, method="near")
  diff_raster <- raster_50m - raster_25m_resampled
  
  diff_df <- as.data.frame(diff_raster * 100, xy = TRUE)
  names(diff_df)[3] <- "difference_cm"
  diff_df$facet_group <- paste("Canopy (Smoothing radius:", sc_val, "m)")
  canopy_diff_df_list[[as.character(sc_val)]] <- diff_df
}
canopy_diff_df <- do.call(rbind, canopy_diff_df_list)

# Prepare DTM difference data using the first iteration's DTMs
rep_sc_val <- subcircle_values[1]
dtm_50m <- dtms_list[[as.character(rep_sc_val)]]$dtm_50m
dtm_25m <- dtms_list[[as.character(rep_sc_val)]]$dtm_25m
dtm_25m_resampled <- resample(dtm_25m, dtm_50m, method="near")
dtm_diff_raster <- dtm_50m - dtm_25m_resampled

dtm_diff_df <- as.data.frame(dtm_diff_raster * 100, xy = TRUE)
names(dtm_diff_df)[3] <- "difference_cm"
dtm_diff_df$facet_group <- "DTM Difference"

# Combine all data for plotting
all_diff_df <- rbind(canopy_diff_df, dtm_diff_df)
all_diff_df <- na.omit(all_diff_df)

# Set the factor levels to control plot order
all_diff_df$facet_group <- factor(
  all_diff_df$facet_group,
  levels = c(
    paste("Canopy (Smoothing radius:", subcircle_values, "m)"),
    "DTM Difference"
  )
)

# --- Define dynamic color scale limits for raster plot ---
max_abs_diff <- max(abs(all_diff_df$difference_cm), na.rm = TRUE)
color_limits <- c(-max_abs_diff, max_abs_diff)

# Create the final faceted raster plot
S_Figure_7A <- ggplot(all_diff_df, aes(x = x, y = y, fill = difference_cm*0.01)) +
  geom_raster() +
  facet_wrap(~ facet_group, ncol = 2) +
  scale_fill_gradient2(
    name = "Difference (m)",
    low = "blue",
    mid = "white",
    high = "red",
    midpoint = 0,
    limits = color_limits*0.01
  ) +
  labs(
    title = "A",
    x = "Easting",
    y = "Northing"
  ) +
  theme_classic() +
  coord_equal() +
  theme(
    plot.title = element_text(size = 20),
    legend.position = c(0.99, 0.55),
    legend.justification = c("right", "bottom"),
    legend.title = element_text(size=10),
    legend.text = element_text(size = 8),
    legend.background = element_blank(),
    legend.box.background = element_rect(colour = "black", fill = NA),
    legend.key.height = unit(0.5, 'cm'),
    strip.text = element_text(size = 14),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12))

print(S_Figure_7A)

# --- Create Faceted Histograms of Differences ---
cat("\n--- Creating Histograms of Differences ---\n")

S_Figure_7B <- ggplot(all_diff_df, aes(x = difference_cm*0.01)) +
  geom_histogram(bins = 50, fill = "grey", color = "black") +
  facet_wrap(~ facet_group, ncol = 2, scales = "free_y") +
  labs(
    title = "B",
    x = "Difference (m)",
    y = "Raster cell frequency"
  ) +
  theme_classic() +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
  theme(plot.title = element_text(size = 20),
        strip.text = element_text(size = 14),
        axis.title = element_text(size = 14),
        axis.text = element_text(size = 12))

print(S_Figure_7B)


# Supplementary Figure 7
#S_Figure_7 <- S_Figure_7A + S_Figure_7B
#print(S_Figure_7)
#ggsave("S_Figure_7.png", plot = S_Figure_7, width = 24, height = 16, dpi = 300)


# Differences in canopy raster cells between elevation and smoothing window

# Prepare data for the coverage plot
coverage_df_list <- list()
for (sc_val in subcircle_values) {
  # Get canopy rasters for the current subcircle value
  raster_50m <- rasters_list[[as.character(sc_val)]]$raster_50m
  raster_25m <- rasters_list[[as.character(sc_val)]]$raster_25m
  
  # Get the DTMs to re-create the common area mask
  dtm_50m <- dtms_list[[as.character(sc_val)]]$dtm_50m
  dtm_25m <- dtms_list[[as.character(sc_val)]]$dtm_25m
  
  # Re-create the overlap mask to constrain the view to the common area
  mask_reg <- !is.na(dtm_50m)
  mask_25_dtm_resampled <- resample(!is.na(dtm_25m), mask_reg, method="near")
  overlap_mask <- mask_reg & mask_25_dtm_resampled
  values(overlap_mask)[!values(overlap_mask)] <- NA
  
  # Resample 25m canopy raster for comparison
  raster_25m_resampled <- resample(raster_25m, raster_50m, method="near")
  
  # Create the coverage raster
  coverage_raster <- raster_50m
  values(coverage_raster) <- NA # Start with a clean slate
  values(coverage_raster)[!is.na(values(raster_50m)) & !is.na(values(raster_25m_resampled))] <- 1  # Both
  values(coverage_raster)[!is.na(values(raster_50m)) & is.na(values(raster_25m_resampled))] <- 2   # 50m only
  values(coverage_raster)[is.na(values(raster_50m)) & !is.na(values(raster_25m_resampled))] <- 3   # 25m only
  
  # Apply the overlap mask to the coverage raster
  coverage_raster_masked <- mask(coverage_raster, overlap_mask)
  
  # Convert to data frame
  coverage_df <- as.data.frame(coverage_raster_masked, xy = TRUE)
  names(coverage_df)[3] <- "coverage_code"
  coverage_df$subcircle <- sc_val
  coverage_df_list[[as.character(sc_val)]] <- coverage_df
}

# Combine all coverage data into one data frame
all_coverage_df <- do.call(rbind, coverage_df_list)
all_coverage_df <- na.omit(all_coverage_df)

# Convert coverage code to a labeled factor for plotting
all_coverage_df$coverage_category <- factor(
  all_coverage_df$coverage_code,
  levels = c(1, 2, 3),
  labels = c("Both", "50m only", "25m only")
)

# Create the faceted coverage plot
coverage_plot <- ggplot(all_coverage_df, aes(x = x, y = y, fill = coverage_category)) +
  geom_raster() +
  facet_wrap(~ subcircle, labeller = label_bquote(rows = "Smoothing radius: " ~ .(subcircle) ~ "m")) +
  scale_fill_manual(
    name = "Data Coverage",
    values = c("Both" = "yellow", "50m only" = "red", "25m only" = "blue")
  ) +
  labs(
    title = "Raster Cell Data Coverage Comparison",
    x = "Easting",
    y = "Northing"
  ) +
  theme_classic() +
  coord_equal()
    
# --- Create Faceted Coverage Plot ---
cat("\n--- Creating Coverage Plot ---\n")

# Prepare data for the coverage plot
coverage_df_list <- list()
for (sc_val in subcircle_values) {
    # Get canopy rasters for the current subcircle value
    raster_50m <- rasters_list[[as.character(sc_val)]]$raster_50m
    raster_25m <- rasters_list[[as.character(sc_val)]]$raster_25m

    # Get the DTMs to re-create the common area mask
    dtm_50m <- dtms_list[[as.character(sc_val)]]$dtm_50m
    dtm_25m <- dtms_list[[as.character(sc_val)]]$dtm_25m
    
    # Re-create the overlap mask to constrain the view to the common area
    mask_reg <- !is.na(dtm_50m)
    mask_25_dtm_resampled <- resample(!is.na(dtm_25m), mask_reg, method="near")
    overlap_mask <- mask_reg & mask_25_dtm_resampled
    values(overlap_mask)[!values(overlap_mask)] <- NA
    
    # Resample 25m canopy raster for comparison
    raster_25m_resampled <- resample(raster_25m, raster_50m, method="near")
    
    # Create the coverage raster
    coverage_raster <- raster_50m
    values(coverage_raster) <- NA # Start with a clean slate
    values(coverage_raster)[!is.na(values(raster_50m)) & !is.na(values(raster_25m_resampled))] <- 1  # Both
    values(coverage_raster)[!is.na(values(raster_50m)) & is.na(values(raster_25m_resampled))] <- 2   # 50m only
    values(coverage_raster)[is.na(values(raster_50m)) & !is.na(values(raster_25m_resampled))] <- 3   # 25m only
    
    # Apply the overlap mask to the coverage raster
    coverage_raster_masked <- mask(coverage_raster, overlap_mask)

    # Convert to data frame
    coverage_df <- as.data.frame(coverage_raster_masked, xy = TRUE)
    names(coverage_df)[3] <- "coverage_code"
    coverage_df$subcircle <- sc_val
    coverage_df_list[[as.character(sc_val)]] <- coverage_df
}

# Combine all coverage data into one data frame
all_coverage_df <- do.call(rbind, coverage_df_list)
all_coverage_df <- na.omit(all_coverage_df)

# Convert coverage code to a labeled factor for plotting
all_coverage_df$coverage_category <- factor(
  all_coverage_df$coverage_code,
  levels = c(1, 2, 3),
  labels = c("Both", "50m only", "25m only")
)

# Create the faceted coverage plot
S_Figure_8A <- ggplot(all_coverage_df, aes(x = x, y = y, fill = coverage_category)) +
  geom_raster() +
  facet_wrap(~ subcircle, labeller = label_bquote(rows = "Smoothing radius: " ~ .(subcircle) ~ "m")) +
  scale_fill_manual(
    name = "Data Coverage",
    values = c("Both" = "yellow", "50m only" = "red", "25m only" = "blue")
  ) +
  labs(
    x = "Easting",
    y = "Northing",
    title = "A"
  ) +
  theme_classic() +
  coord_equal() +
  theme(legend.position = c(0.8, 0.20),
        legend.justification = c("right", "bottom"),
        strip.text = element_text(size = 14),
        axis.title = element_text(size = 14),
        plot.title = element_text(size = 20))

print(S_Figure_8A)

# Barplot of cells
# Summarize counts by coverage category (and subcircle, if desired)
coverage_counts <- all_coverage_df %>%
  group_by(subcircle, coverage_category) %>%
  summarise(cell_count = n()) %>%
  ungroup()

S_Figure_8B <- ggplot(coverage_counts, aes(x = coverage_category, y = cell_count, fill = coverage_category)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~ subcircle, labeller = label_bquote("Smoothing radius: " ~ .(subcircle) ~ "m")) +
  scale_fill_manual(
    name = "Data Coverage",
    values = c("Both" = "yellow", "50m only" = "red", "25m only" = "blue")
  ) +
  labs(
    x = "",
    y = "Number of Cells",
    title = "B",
  ) +
  theme_classic() +
  theme(
    legend.position = "none",
    strip.text = element_text(size = 14),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(size = 20))

print(S_Figure_8B)

#S_Figure_8 <- S_Figure_8A + S_Figure_8B
#print(S_Figure_8)
#ggsave("S_Figure_8.png", plot = S_Figure_8, width = 16, height = 8, dpi = 300)




#######################################
#######################################
#######################################
#######################################
#######################################


# Comparison of LiDAR with handheld RTK GNSS elevation and Otter bathymetry

# elevation shapefile, created in ARC GIS
elevation <- st_read("RS3_elevation_GT.shp")

crs(idw_smoothed)
crs(idw_smoothed_25)
crs(elevation)

# extracting elevation values of points underwater

# Convert the sf object to a SpatVector (terra format)
elevation_vect <- vect(elevation)

# Extract raster values at the point locations
raster_values_50 <- terra::extract(idw_smoothed, elevation_vect) #focal_mean
raster_values_25 <- terra::extract(idw_smoothed_25, elevation_vect) #focal_mean.1

# Combine the extracted raster values with the original shapefile data
elevation_with_values <- cbind(elevation, raster_values_50, raster_values_25)

# filter to rows with underwater elevation
RS3_underwater <- elevation_with_values %>% drop_na(focal_mean.1)

# Filter out rows where 'Name' contains 't' (case-insensitive)
RS3_underwater_bottom <- RS3_underwater %>%
  filter(!grepl("t", Name, ignore.case = TRUE))

RS3_underwater_bottom <- RS3_underwater_bottom %>% filter(Origin == "Global") %>% filter(Tilt_angle <= 5) #%>% filter(Descriptio == "validation line")
nrow(RS3_underwater_bottom)

# applying antenna correction
RS3_underwater_bottom$Elevation <- RS3_underwater_bottom$Elevation - 0.2

# taking only values from the sandy bottom
sand <- c(-0.749, -0.814, -0.854, -1.035, -1.089, -1.136)


# taking elevation values from June 2023 (local origin rows in data)

# Filter the dataset
RS3_underwater_bottom <- RS3_underwater_bottom %>%
  filter(sapply(Elevation, function(x) any(abs(x - sand) < 1e-3)))

# add local origins! RTK values from June 2023
RS3_underwater_june_2023 <- RS3_underwater %>%
  filter(Origin == "Local")

# join with June 2023 data
RS3_underwater_bottom <- rbind(RS3_underwater_bottom, RS3_underwater_june_2023)

# calculating RS3 vertical and horizontal error
RS3_error <- RS3_underwater_bottom[RS3_underwater_bottom$Easting_RM != 0, ]
erm <- mean(RS3_error$Easting_RM)
nrm <- mean(RS3_error$Northing_R)
evrm <- mean(RS3_error$Elevation_)



# LiDAR bottom point elevation versus Rs3
# Ensure CRS matches
print(st_crs(RS3_underwater_bottom))
print(st_crs(sea_bottom_las))
print(st_crs(sea_bottom_las_25))

# Extract Z-values from the bottom at GT point locations
gt_coords_sand <- st_coordinates(RS3_underwater_bottom)

# Find k nearest neighbors using nn2
k <- 1

# Convert bottom LAS point cloud into a dataframe for NN analysis
ele_df <- data.frame(
  X = sea_bottom_las@data$X,
  Y = sea_bottom_las@data$Y,
  Z = sea_bottom_las@data$Z
)

ele_df_25 <- data.frame(
  X = sea_bottom_las_25@data$X,
  Y = sea_bottom_las_25@data$Y,
  Z = sea_bottom_las_25@data$Z
)

nn <- nn2(data = ele_df[, c("X", "Y", "Z")], query = gt_coords_sand, k = k)
nn_25 <- nn2(data = ele_df_25[, c("X", "Y", "Z")], query = gt_coords_sand, k = k)

# Extract Z-values for the nearest neighbors
neighbor_indices <- nn$nn.idx
z_values <- apply(neighbor_indices, 1, function(indices) ele_df$Z[indices])

neighbor_indices_25 <- nn_25$nn.idx
z_values_25 <- apply(neighbor_indices_25, 1, function(indices) ele_df_25$Z[indices])

RS3_underwater_bottom$Z_mean <- z_values
RS3_underwater_bottom$Z_mean_25 <- z_values_25


# Plot LiDAR vs RS3 bottom elevation
LiDAR_RS3 <- data.frame(
  Validation_data = RS3_underwater_bottom$Elevation,
  Elevation_50 = RS3_underwater_bottom$Z_mean,
  Elevation_25 = RS3_underwater_bottom$Z_mean_25) %>%
  mutate(data = "LiDAR ~ Handheld GNSS")


LiDAR_RS3 <- LiDAR_RS3 %>%
  pivot_longer(
    cols = c(Elevation_50, Elevation_25),
    names_to = "Dataset",
    values_to = "Elevation"
  ) %>%
  mutate(Dataset = recode(Dataset, "Elevation_50" = "50 m", "Elevation_25" = "25 m"))

LiDAR_RS3 <- LiDAR_RS3 %>%
  select(Validation_data, Elevation, Dataset, data)








# Plot raster vs canopy heights, IDW DTM and P2R DSM
plot_data_DTM_elev <- data.frame(
  RS3 = RS3_underwater_bottom$Elevation,
  IDW_50 = RS3_underwater_bottom$focal_mean,
  IDW_25 = RS3_underwater_bottom$focal_mean.1)

# Fit linear models
lm_fit_DTM_50_RS3 <- lm(IDW_50 ~ RS3, data = plot_data_DTM_elev)
summary(lm_fit_DTM_50_RS3)

lm_fit_DTM_25_RS3 <- lm(IDW_25 ~ RS3, data = plot_data_DTM_elev)
summary(lm_fit_DTM_25_RS3)


# Extract R squared value
r_squared_50 <- summary(lm_fit_DTM_50_RS3)$r.squared
r_squared_25 <- summary(lm_fit_DTM_25_RS3)$r.squared

plot_data_DTM_elev$data <- c("LiDAR ~ Handheld GNSS")

DTM_elev <- plot_data_DTM_elev %>%
  pivot_longer(
    cols = c(IDW_50, IDW_25),
    names_to = "Dataset",
    values_to = "IDW"
  ) %>%
  mutate(Dataset = recode(Dataset, "IDW_50" = "50 m", "IDW_25" = "25 m"))

DTM_elev <- DTM_elev %>%
  select(RS3, IDW, Dataset, data)

names(DTM_elev) <- c("Validation_data", "Elevation", "Dataset", "data")


# Loading full LiDAR datasets, to compare elevation outside of water area
full_50 <- readLAS("../Olberg_20240829_50m_reproc_211124.laz")
full_50 <- fix_crs_encoding(full_50)
full_25 <- readLAS("Olberg_25m_reproc_211124.las")
full_25 <- fix_crs_encoding(full_25)

# LiDAR point elevation versus Rs3
# Ensure CRS matches
print(st_crs(elevation))
print(st_crs(full_50))
print(st_crs(full_25))

elevation$Code <- as.factor(elevation$Code)
str(elevation)


# Here, we try correcting data gathered
# with incorrect antenna height
elevation_land <- elevation %>%
  filter(Code == "VL" | Code == "L_3") %>%
  filter(Elevation > 0) %>%
  filter(Tilt_angle < 5) %>%
  mutate(Elevation = case_when(
    Descriptio == "validation line" ~ Elevation + 0.2,
    TRUE ~ Elevation
  ))



##### only using above-ground points that don't require correction
elevation_land <- elevation %>%
  filter(Code == "VL" | Code == "L_3") %>%
  filter(Elevation > 0)

elevation_land <- elevation_land[!grepl("validation line", elevation_land$Descriptio), ]



gt_coords <- st_coordinates(elevation_land)

# Find nearest neighbor(s) using nn2
k <- 1

# 50 m data
# Convert LAS point cloud into a dataframe for NN analysis
full_df_50 <- data.frame(
  X = full_50@data$X,
  Y = full_50@data$Y,
  Z = full_50@data$Z
)
nn_50 <- nn2(data = full_df_50[, c("X", "Y", "Z")], query = gt_coords, k = k)
# Extract Z-values for the nearest neighbor(s)
neighbor_indices <- nn_50$nn.idx
z_values_50 <- apply(neighbor_indices, 1, function(indices) full_df_50$Z[indices])
elevation_land$z_land_mean_50 <- z_values_50

# 25 m data
# Convert LAS point cloud into a dataframe for NN analysis
full_df_25 <- data.frame(
  X = full_25@data$X,
  Y = full_25@data$Y,
  Z = full_25@data$Z
)
nn_25 <- nn2(data = full_df_25[, c("X", "Y", "Z")], query = gt_coords, k = k)
# Extract Z-values for the nearest neighbor(s)
neighbor_indices <- nn_25$nn.idx
z_values_25 <- apply(neighbor_indices, 1, function(indices) full_df_25$Z[indices])
elevation_land$z_land_mean_25 <- z_values_25




# Plot terrestrial LiDAR vs RS3
plot_data_LiDAR_RS3 <- data.frame(
  RS3 = elevation_land$Elevation,
  LiDAR_50 = elevation_land$z_land_mean_50,
  LiDAR_25 = elevation_land$z_land_mean_25)

# Fit linear models
# 50 m
lm_fit_terr_50 <- lm(LiDAR_50 ~ RS3, data = plot_data_LiDAR_RS3)
summary(lm_fit_terr_50)
r_squared_terr_50 <- summary(lm_fit_terr_50)$r.squared
r_squared_terr_50 

# 25 m
lm_fit_terr_25 <- lm(LiDAR_25 ~ RS3, data = plot_data_LiDAR_RS3)
summary(lm_fit_terr_25)
r_squared_terr_25 <- summary(lm_fit_terr_25)$r.squared
r_squared_terr_25 

above_ground_comparison <- plot_data_LiDAR_RS3 %>%
  pivot_longer(
    cols = c(LiDAR_50, LiDAR_25),
    names_to = "Dataset",
    values_to = "LiDAR"
  ) %>%
  mutate(Dataset = recode(Dataset, 
                          "LiDAR_50" = "50 m", 
                          "LiDAR_25" = "25 m"))


# Plot
S_Figure_4 <- ggplot(above_ground_comparison, aes(x = RS3, y = LiDAR, color = Dataset, shape = Dataset)) +
  geom_point(size = 2) +
  geom_smooth(method = "lm", se = TRUE) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +
  scale_color_manual(values = c("25 m" = "sienna2", "50 m" = "slategrey")) +
  scale_shape_manual(values = c("25 m" = 16, "50 m" = 17)) +
  scale_x_continuous(limits = c(0, 8), breaks = seq(0, 8, by = 1)) +
  scale_y_continuous(limits = c(0, 8), breaks = seq(0, 8, by = 1)) +
  labs(
    x = "Ground-truth Elevation (m)", 
    y = "LiDAR Point Elevation (m)") +
  theme_classic() +
  theme(
    axis.text = element_text(size = 18),
    axis.title = element_text(size = 20),
    legend.text = element_text(size = 12),
    legend.title = element_text(size = 14),
    legend.position = c(0.7, 0.32))

print(S_Figure_4)
#ggsave("S_Figure_4.png", plot = S_Figure_4, device = "png", width = 8, height = 6, dpi = 300)

terrain_error <- above_ground_comparison %>%
  group_by(Dataset) %>%
  mutate(error = RS3 - LiDAR) %>%
  summarize(mean_error = mean(error, na.rm = T)*100,
            sd_error = sd(error, na.rm = T)*100,
            rmse = rmse(RS3, LiDAR)*100) %>%
  ungroup()


print(terrain_error)



# Comparing LiDAR bathy with otter bathy
otter <- read.csv("2023-09-13_084848_TideCorrectedBathymetryRel2NN2000.csv", sep = ";")

# Convert to sf object
otter_sf <- st_as_sf(otter, coords = c("Longitude_deg", "Latitude_deg"), crs = 4326)

target_crs <- crs(DSM_final_cropped)

# Transform coordinates to the target CRS
otter_transformed <- st_transform(otter_sf, crs = target_crs)

# Extract transformed coordinates back into a dataframe
otter$X_utm <- st_coordinates(otter_transformed)[,1]
otter$Y_utm <- st_coordinates(otter_transformed)[,2]

# View transformed data
head(otter)
head(sea_bottom_points)
head(sea_bottom_points_25)

# extracting nearest points between otter and LiDAR

# Extract coordinates from both datasets
otter_coords <- otter[, c("X_utm", "Y_utm")]
sea_bottom_coords <- as.data.frame(sea_bottom_points[, 1:2])
sea_bottom_coords_25 <- as.data.frame(sea_bottom_points_25[, 1:2])
colnames(sea_bottom_coords) <- c("X_utm", "Y_utm")
colnames(sea_bottom_coords_25) <- c("X_utm", "Y_utm")

# Find the nearest neighbor in `sea_bottom_points` for each `otter` point
nn_result <- nn2(data = sea_bottom_coords, query = otter_coords, k = 1)
nn_result_25 <- nn2(data = sea_bottom_coords_25, query = otter_coords, k = 1)

# Extract indices of nearest points
nearest_indices <- nn_result$nn.idx
nearest_indices_25 <- nn_result_25$nn.idx

# Add nearest X, Y, distance, and Z to `otter`
otter$nearest_X <- sea_bottom_coords[nearest_indices, "X_utm"]
otter$nearest_Y <- sea_bottom_coords[nearest_indices, "Y_utm"]
otter$distance_to_nearest <- nn_result$nn.dists
otter$nearest_Z <- sea_bottom_points[nearest_indices, 3]

otter$nearest_X_25 <- sea_bottom_coords_25[nearest_indices_25, "X_utm"]
otter$nearest_Y_25 <- sea_bottom_coords_25[nearest_indices_25, "Y_utm"]
otter$distance_to_nearest_25 <- nn_result_25$nn.dists
otter$nearest_Z_25 <- sea_bottom_points_25[nearest_indices_25, 3]

# View the updated `otter` dataframe
otter_overlap <- otter %>% dplyr::filter(distance_to_nearest < 0.1)
#otter_overlap$nearest_Z <- otter_overlap$nearest_Z * -1

# adding 45 cm of depth to otter Z (accounting for sensor position / tide)
otter_overlap$Z <- otter_overlap$bottomElevRel2NN2000

# removing points below 70 cm LiDAR depth
otter_overlap <- otter_overlap %>% filter(nearest_Z < -1.1)

# Plot
ggplot(otter_overlap, aes(x = Z, y = nearest_Z_25)) +
  geom_point(color = "blue", size = 2) +
  geom_smooth(method = "lm", color = "red", se = TRUE) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +
  #scale_x_continuous(limits = c(0, 3), breaks = seq(0, 3, by = 0.25)) +
  #scale_y_continuous(limits = c(0, 3), breaks = seq(0, 3, by = 0.25)) +
  labs(
    x = "Otter depth (m)", 
    y = "LiDAR depth (m)") +
  theme_classic() +
  theme(
    axis.text = element_text(size = 18),
    axis.title = element_text(size = 20),
    legend.text = element_text(size = 12),
    legend.title = element_text(size = 14))


# Creating dataframe for Figure 5
head(LiDAR_RS3)
head(otter_overlap)

otter_compare <- otter_overlap %>% select(Z, nearest_Z, nearest_Z_25) %>%
  mutate(data = c("LiDAR ~ Echosounder"))

otter_compare <- otter_compare %>%
  pivot_longer(
    cols = c(nearest_Z, nearest_Z_25),
    names_to = "Dataset",
    values_to = "Elevation"
  ) %>%
  mutate(Dataset = recode(Dataset, "nearest_Z" = "50 m", "nearest_Z_25" = "25 m"))

otter_compare <- otter_compare %>%
  select(Z, Elevation, Dataset, data)

names(otter_compare) <- c("Validation_data", "Elevation", "Dataset", "data")


# join RS3 and Otter data
LiDAR_relationship <- rbind(LiDAR_RS3, otter_compare)


Figure_5 <- ggplot(LiDAR_relationship, aes(x = Validation_data, y = Elevation, color = data)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE) +
  facet_wrap(~Dataset, ncol = 1) +
  labs(
    x = "Ground-truth sea floor elevation (m)",
    y = "LiDAR sea floor elevation (m)",
    color = "Data Sources"   # <-- Change legend title here
  ) +
  scale_x_continuous(limits = c(-2.75, -0.50), breaks = seq(-2.75, -0.50, by = 0.25)) +
  scale_y_continuous(limits = c(-2.75, -0.50), breaks = seq(-2.75, -0.50, by = 0.25)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +
  theme_classic() +
  theme(
    axis.text = element_text(size = 18),
    axis.title = element_text(size = 20),
    plot.title = element_text(size = 20, hjust = 0.5),
    legend.text = element_text(size = 12),
    legend.title = element_text(size = 14),
    legend.position = c(0.01, 0.99),
    legend.justification = c("left", "top"),
    strip.text = element_text(size = 16)) +
  scale_color_manual(
    values = c(
      "LiDAR ~ Handheld GNSS" = "blue",
      "LiDAR ~ Echosounder" = "red"),
    breaks = c("LiDAR ~ Handheld GNSS", "LiDAR ~ Echosounder"))


Figure_5

# Save the arranged figure as a .tif file
#ggsave("Figure_5_new.png", plot = Figure_5, width = 8, height = 14, dpi = 300)

# Calculating R-squared values for each linear model
r_squared_results <- LiDAR_relationship %>%
  group_by(Dataset, data) %>%
  summarize(
    model_fit = list(lm(Elevation ~ Validation_data)),
    .groups = "drop"
  ) %>%
  mutate(
    r_squared = sapply(model_fit, function(m) summary(m)$r.squared),
    adj_r_squared = sapply(model_fit, function(m) summary(m)$adj.r.squared),
    intercept = sapply(model_fit, function(m) coef(m)[1]),
    slope = sapply(model_fit, function(m) coef(m)[2])
  ) %>%
  select(-model_fit)

print(r_squared_results)
cat("\n")


# Calculating vertical errors
LiDAR_relationship$data <- as.factor(LiDAR_relationship$data)
LiDAR_relationship$Dataset <- as.factor(LiDAR_relationship$Dataset)

error_summary <- LiDAR_relationship %>%
  mutate(error = (Elevation - Validation_data) * 100) %>%  # error in cm
  group_by(Dataset, data) %>%
  summarise(
    mean_error = mean(error, na.rm = TRUE),
    sd_error = sd(error, na.rm = TRUE),
    rmse = sqrt(mean(error^2, na.rm = TRUE)),
    n = n()
  ) %>%
  ungroup()

print(error_summary)





# plotting over points
# Extract X, Y, and Z values
x <- study_area_points[,1]  # UTM X
y <- study_area_points[,2]  # UTM Y
z <- study_area_points[,3]  # Depth/Fill value

# Define a color palette
#library(viridis)
col_pal <- colorRampPalette(c("black","gray40","gray70","gray90"))(100)
col_vals <- cut(z, breaks = 100, labels = FALSE)

# Plot the points with color scale
plot(x, y, col = col_pal[col_vals], pch = 16, cex = 0.2,
     xlab = "Easting (m)", ylab = "Northing (m)")
points(otter_overlap$X_utm, otter_overlap$Y_utm, col = "red", pch = 16)

# plotting as raster
res_value <- 0.3  # increasing raster size for visualisation

# Create an empty raster with desired resolution
r <- rast(xmin = min(x), xmax = max(x), ymin = min(y), ymax = max(y),
          resolution = res_value)

# Assign Z values to raster cells (using mean where multiple points exist)
r <- rasterize(cbind(x, y), r, z, fun = mean)





# Plot raster with scale bar
# Set up the PNG file to save the plot
png("C:/Users/seabee/OneDrive - NIVA/LiDAR/Charlie_Ohlberg/SeaBee LiDAR ZOSMA 2024/R/S_Figure_1_new.png", 
    width = 8, height = 8, units = "in", res = 200)

# Set margins and outer margins inside the PNG device
par(mar = c(0, 0, 0, 2))  # Adjust margins: bottom, left, top, right
par(oma = c(0, 0, 0, 2))  # Adjust outer margins for more space

# Plot the raster
plot(r, col = col_pal,
     xlab = "Eastings",
     ylab = "Northings")

# Add otter points
points(otter_overlap$X_utm, otter_overlap$Y_utm, col = "red", pch = 16, cex = 0.65)

# Define legend position (manually placing in top-right)
legend_x <- max(x) - 70
legend_y <- max(y) - c(10, 17, 24)  # Y positions for three rows

# Add points for legend
points(rep(legend_x - 5, 3), legend_y, col = c("red", "blue", "green"), pch = 16, cex = 1.5)

# Add text labels next to points
text(legend_x, legend_y, labels = c("Echosounder bathymetry", "Handheld RTK GNSS depth", "In-situ canopy height"), 
     col = c("red", "blue", "green"), cex = 1, pos = 4)

# Add rotated text grobs for the labels
grid.text("LiDAR bathymetry (m)", x = unit(0.96, "npc"), y = unit(0.5, "npc"), rot = -90, gp = gpar(fontsize = 12))

# Add points from gt_points and RS3_underwater_bottom (assuming they're in the correct CRS)
set.seed(42)  # For reproducibility
jitter_amount <- 0.5  # Adjust as needed
points(st_coordinates(gt_points) + runif(nrow(gt_points), -jitter_amount, jitter_amount), 
       col = "green", pch = 16, cex = 0.7)

points(st_coordinates(RS3_underwater_bottom), col = "blue", pch = 16, cex = 0.7)

# Close the PNG device (this will save the plot)
dev.off()
