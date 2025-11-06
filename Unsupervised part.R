# Install packages
install.packages(c("dplyr", "ggplot2", "cluster", "factoextra"))
install.packages("ggcorrplot")


# Load required libraries
library(dplyr)
library(ggplot2)
library(cluster)
library(factoextra)
library(gridExtra)
library(ggcorrplot)
library(scales)


#Loading Data
unsupervised <- read.csv("F:/University/Projects/Data Science/Statistical Learning/Maven Project/unsupervised.csv")

###Data preparation
unsupervised$row_id <- seq_len(nrow(unsupervised))
str(unsupervised)

# Convert 'became_member_on' to Date type
unsupervised$became_member_on <- as.Date(unsupervised$became_member_on)


# Creating a feature for Membership Duration (in days)
unsupervised$membership_duration <- as.numeric(as.Date("2024-09-15") - unsupervised$became_member_on)


# Encode 'gender' as numeric
unsupervised$gender <- as.numeric(factor(unsupervised$gender, levels = c("M", "F", "O")))

# Normalize the data (excluding non-numeric columns)
data_numeric <- unsupervised %>%
  select(where(is.numeric)) %>%
  scale()
# Convert to data frame
data_numeric <- as.data.frame(data_numeric)



###Exploratory Data Analysis
# Visualize the data to check for outliers
boxplot(data_numeric, las = 2)

# Function to check if a value is an outlier
is_outlier <- function(column) {
  Q1 <- quantile(column, 0.25)
  Q3 <- quantile(column, 0.75)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  return(column < lower_bound | column > upper_bound)
}

# Initialize a logical vector to track rows with outliers
rows_with_outliers <- rep(FALSE, nrow(data_numeric))

# Loop through each column and mark rows containing outliers
for (i in seq_len(ncol(data_numeric))) {
  outlier_flags <- is_outlier(data_numeric[, i])
  rows_with_outliers <- rows_with_outliers | outlier_flags  # Update for any row with an outlier
}

# Remove rows with outliers
keep_idx <- which(!rows_with_outliers)
data_cleaned <- data_numeric[keep_idx, , drop = FALSE]
kept_row_ids <- unsupervised$row_id[keep_idx]

data <- as.data.frame(data_cleaned)

# Print the result
cat("Original number of rows:", nrow(data_numeric), "\n")
cat("Number of rows after removing outliers:", nrow(data_cleaned), "\n")
print(data_cleaned)

# Visualize the data to check for outliers
boxplot(data_cleaned, las = 2)

# Correlation matrix

corr_matrix <- cor(data_cleaned)

ggcorrplot(corr_matrix, type = "lower", outline.color = "white", lab = TRUE,
           colors = c("darkred","#FFFFE0","darkblue")) +
  labs(title = "Correlation Heatmap")


# Check column means and standard deviations
colMeans(data_cleaned)
apply(data_cleaned,2, sd)

###PCA
# Execute PCA, with scaling: customer.pr
customer.pr <- prcomp(data_cleaned, scale. = FALSE)

summary(customer.pr)

# Create a biplot of customer.pr
biplot(customer.pr)

# Calculate variability of each component
pr.var <- customer.pr$sdev ^2

# Variance explained by each principal component: pve
pve <- pr.var/sum(pr.var)

# Plot the bar plot for individual component variance
plot(pve, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained", ylim = c(0, 1), type = "h",
     lwd = 10, col = "skyblue", xaxt = 'n')

# Add x-axis labels
axis(1, at = 1:length(pve), labels = 1:length(pve))

# Add cumulative variance line to the same plot
lines(cumsum(pve), type = "b", pch = 16, col = "red", lwd = 2)

# Add a legend
legend("right", legend = c("Individual Variance", "Cumulative Variance"),
       col = c("skyblue", "red"), lty = c(1, 1), lwd = 2, pch = c(NA, 16))

###HCust
# Calculate the (Euclidean) distances: data.dist
data.dist <- dist(data_cleaned)

# Create a hierarchical clustering model: customer.hclust
customer.hclust <- hclust(data.dist, method = 'complete')

# Plot with abbreviated labels
plot(customer.hclust, main = "Hierarchical Clustering Dendrogram")

# Add a horizontal cut line at the appropriate height
abline(h = 8.5, col = "red", lty = 2)

# Add rectangles to highlight the clusters
rect.hclust(customer.hclust, k = 6, border = 2:5)

###KMeans
##Elbow Method
# Compute K-Means clustering for a range of k
wss <- sapply(1:10, function(k) {
  kmeans(data_cleaned, centers = k, nstart = 25)$tot.withinss
})

# Plot elbow method
plot(1:10, wss, type = "b", xlab = "Number of clusters (k)", ylab = "Within-cluster sum of squares")

# Set number of clusters
k <- 6

# Apply K-Means clustering
set.seed(123) # For reproducibility
kmeans_result <- kmeans(data_cleaned, centers = k, nstart = 25)

# Add cluster assignment to the original data
data$cluster <- kmeans_result$cluster

# View clustering result
table(data$cluster)

data_pca <- as.data.frame(customer.pr$x)
# Add cluster assignment
data_pca$cluster <- as.factor(data$cluster)

summary(customer.pr)
# Plot the clusters
ggplot(data_pca, aes(x = PC1, y = PC2, color = cluster)) +
  geom_point() +
  labs(title = "Clustering Visualization with PCA")


### PCA for Clustering
# We already performed PCA: customer.pr
# Extract the principal components
pca_data <- customer.pr$x[, 1:6]

### K-Means Clustering with PCA
## Elbow Method on PCA data
# Compute K-Means clustering for a range of k
wss_pca <- sapply(1:10, function(k) {
  kmeans(pca_data, centers = k, nstart = 25)$tot.withinss
})

# Plot elbow method to determine optimal k
plot(1:10, wss_pca, type = "b", xlab = "Number of clusters (k)", ylab = "Within-cluster sum of squares")

# Set number of clusters
k <- 6

# Apply K-Means clustering on PCA data
set.seed(123)
kmeans_result_pca <- kmeans(pca_data, centers = k, nstart = 25)

# Add cluster assignment to PCA data
data_pca$cluster <- as.factor(kmeans_result_pca$cluster)

cluster_map <- data.frame(
  row_id  = kept_row_ids,
  cluster = factor(kmeans_result_pca$cluster)
)

df <- unsupervised %>%
  inner_join(cluster_map, by = "row_id")


x_visits <- "num_transactions"                 # تعداد تراکنش‌ها (visits)
y_basket <- "avg_spending_per_transaction"     # میانگین مبلغ هر تراکنش (basket)
rev_col  <- "total_spent"                      # مجموع هزینه/درآمد دوره (revenue)
age_col  <- "age"                              # سن
inc_col  <- "income"                           # درآمد


### Plot the clusters using the PCA results
ggplot(data_pca, aes(x = PC2, y = PC1, color = cluster)) +
  geom_point() +
  labs(title = "K-Means Clustering Visualization with PCA")

# View clustering result
table(data_pca$cluster)


### Hierarchical Clustering with PCA

# Calculate the Euclidean distances on the PCA data
pca_dist <- dist(pca_data)

# Create a hierarchical clustering model on the PCA data
customer_hclust_pca <- hclust(pca_dist, method = 'complete')

# Plot the dendrogram
plot(customer_hclust_pca, main = "Hierarchical Clustering Dendrogram (PCA)", xlab = "", sub = "", cex = 0.9)

# Optionally, you can cut the tree to form clusters
# Cut the dendrogram into 6 clusters
rect.hclust(customer_hclust_pca, k = 6, border = 2:5)

# Add a horizontal cut line at the appropriate height for 6 clusters
abline(h = 8.5, col = "red", lty = 2)

# Assign cluster memberships
pca_hclust_clusters <- cutree(customer_hclust_pca, k = 6)

# Add the cluster assignments to the PCA data
data_pca$hclust_cluster <- as.factor(pca_hclust_clusters)

### Visualization of the clusters using the PCA results
ggplot(data_pca, aes(x = PC1, y = PC2, color = hclust_cluster)) +
  geom_point() +
  labs(title = "Hierarchical Clustering Visualization with PCA")




cluster_profile <- df %>%
  dplyr::group_by(cluster) %>%
  dplyr::summarise(
    n_customers         = dplyr::n(),
    share_of_customers  = n_customers / nrow(df),

    revenue_total       = sum(.data[["total_spent"]], na.rm = TRUE),
    share_of_revenue    = revenue_total / sum(df[["total_spent"]], na.rm = TRUE),

    # Visits (num_transactions)
    visits_med          = median(.data[["num_transactions"]], na.rm = TRUE),
    visits_mean         = mean(.data[["num_transactions"]], na.rm = TRUE),
    visits_iqr          = IQR(.data[["num_transactions"]], na.rm = TRUE),
    visits_sd           = sd(.data[["num_transactions"]],  na.rm = TRUE),

    # Basket (avg_spending_per_transaction)
    basket_med          = median(.data[["avg_spending_per_transaction"]], na.rm = TRUE),
    basket_mean         = mean(.data[["avg_spending_per_transaction"]], na.rm = TRUE),
    basket_iqr          = IQR(.data[["avg_spending_per_transaction"]], na.rm = TRUE),
    basket_sd           = sd(.data[["avg_spending_per_transaction"]],  na.rm = TRUE),

    # Two “typical spend” definitions
    monthly_spend_typ_med  = visits_med  * basket_med,   # رفتار تیپیک (robust)
    monthly_spend_typ_mean = visits_mean * basket_mean,  # مقدار مورد انتظار

    age_mean            = mean(.data[["age"]],    na.rm = TRUE),
    income_mean         = mean(.data[["income"]], na.rm = TRUE)
  ) %>%
  dplyr::arrange(dplyr::desc(share_of_revenue))

print(cluster_profile)


cluster_bullets <- cluster_profile %>%
  mutate(
    title = paste0(
      "Cluster ", cluster,
      " — Customers ", percent(share_of_customers, accuracy = 1),
      " | Revenue ", percent(share_of_revenue, accuracy = 1)
    ),
    bullets = paste0(
      "• Size: ", n_customers, " (", percent(share_of_customers, accuracy = 1), " of customers)\n",
      "• Revenue: ", round(revenue_total, 2), " (", percent(share_of_revenue, accuracy = 1), " of total)\n",
      "• Visits — median ", round(visits_med), " (IQR ", round(visits_iqr, 2), "); ",
      "mean ", round(visits_mean, 2), " (sd ", round(visits_sd, 2), ")\n",
      "• Basket — median ", round(basket_med, 2), " (IQR ", round(basket_iqr, 2), "); ",
      "mean ", round(basket_mean, 2), " (sd ", round(basket_sd, 2), ")\n",
      "• Typical spend — median×median ", round(monthly_spend_typ_med, 2),
      "; mean×mean ", round(monthly_spend_typ_mean, 2), "\n",
      ifelse(is.na(age_mean),    "", paste0("• Mean age: ", round(age_mean), "\n")),
      ifelse(is.na(income_mean), "", paste0("• Mean income: ", comma(round(income_mean)), "\n"))
    )
  ) %>%
  arrange(desc(share_of_revenue)) %>%
  select(cluster, title, bullets)

# 8.2 چاپ خوانا در کنسول
invisible(lapply(seq_len(nrow(cluster_bullets)), function(i) {
  cat("\n", cluster_bullets$title[i], "\n", cluster_bullets$bullets[i], sep = "")
}))

