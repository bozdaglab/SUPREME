# # # # # # # # # # # # # # # # # # # # # # # # # 
#  SUPREME- Similarity Network Generation Part  # 
# # # # # # # # # # # # # # # # # # # # # # # # # 

require(cluster)
set.seed(404)
topx = 10 # number of edges to keep 
# If the similarity metrics are same, then it will include all the edges with that value, thus might couse to have a network with more than that number.

# Input df's have the form of (samples x features): that is, rows are samples, columns are features.

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Generation of similarity network with a continuous datatype
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Input: a matrix with continuous values (samples x features)
# Similarity metric: Pearson Correlation
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
df = as.data.frame(matrix(rnorm(10*5,10,3), ncol=5))
c = cor(t(df)); colnames(c) = 0:(ncol(c)-1); rownames(c) = 0:(ncol(c)-1)
diag(c) = 1 # self nodes for missing samples
network = reshape2::melt(c)
network = network[network$value >= sort(network$value, decreasing = TRUE)[topx*2],] # topx*2 is for two-way relations
network$value = 1


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Generation of similarity network with a binary datatype
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Input: a matrix with binary values (samples x features)
# Similarity metric: Jaccard Similarity
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
df = as.data.frame(matrix(rnorm(10*5,0.1,0.5), ncol=5)); df[df > 0] = 1; df[df <= 0] = 0
s <- 1- as.matrix(dist(df, method = "binary"))
c <- matrix(0, ncol=nrow(df), nrow=nrow(df)); rownames(c) = 1:nrow(df); colnames(c) = 1:nrow(df)
c[rownames(s), colnames(s)] <- unlist(s); colnames(c) = 0:(ncol(c)-1); rownames(c) = 0:(ncol(c)-1) 
diag(c) = 1 # self nodes for missing samples
network = reshape2::melt(c)
network = network[network$value >= sort(network$value, decreasing = TRUE)[topx*2],]
network$value = 1


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Generation of similarity network with a mixed datatype (categorical, continuous, and/or binary)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Input: a matrix with binary values (samples x features)
# Similarity metric: Gower Similarity
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
df = as.data.frame(matrix(rnorm(10*5,10,3), ncol=5))
s = 1- as.matrix(daisy(df, metric="gower"))
c <- matrix(0, ncol=nrow(df), nrow=nrow(df)); rownames(c) = 1:nrow(df); colnames(c) = 1:nrow(df)
c[rownames(s), colnames(s)] <- unlist(s); colnames(c) = 0:(ncol(c)-1); rownames(c) = 0:(ncol(c)-1) 
diag(c) = 1 # self nodes for missing samples
network = reshape2::melt(c)
network = network[network$value >= sort(network$value, decreasing = TRUE)[topx*2],]
network$value = 1
