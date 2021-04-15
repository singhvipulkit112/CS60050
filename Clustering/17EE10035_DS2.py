#########################################################################################################################################################
# 17EE10035 : Pulkit Singhvi
# MiniProject3_DS2
# Project Title : Coronavirus Data Clustering using Single Linkage Hierarchical Clustering Technique
#########################################################################################################################################################

import numpy as np
import pandas as pd

np.random.seed(16)

# Function to calc euclidean dist btw two vectors ---------------------------------------------------

def euclidean_dist(x,y):

    return np.sqrt(sum((x - y) ** 2))

# ---------------------------------------------------------------------------------------------------


# K means algorithm ---------------------------------------------------------------------------------

def k_means(X, k):

    # k cluster means randomly initialized as k distinct data points
    C = np.random.choice(range(len(X)), size=k, replace=False)
    C = X[C,:]

    clusters = np.zeros(len(X), dtype=int)

    for iter in range(20):

        for i in range(len(X)):

            min_dist = 123123123.0

            for j in range(k):
                dist_between_i_and_Cj = euclidean_dist(C[j],X[i])
                
                if(dist_between_i_and_Cj < min_dist):
                    min_dist = dist_between_i_and_Cj
                    clusters[i]=j
                    
        groups = [[] for _ in range(k)]
        cluster_blobs = [[] for _ in range(k)]

        for i in range(len(X)):
            groups[clusters[i]].append(X[i])
            cluster_blobs[clusters[i]].append(i)

        for i in range(k):
            if len(groups[i]) > 1:
              C[i]=list(np.mean(groups[i],axis=0))
            else :
              C[i]=list(groups[i])

    return [C, groups, cluster_blobs, clusters]

# ---------------------------------------------------------------------------------------------------


# Function to save the clustering information in a file ---------------------------------------------

def save_clustering_info(cluster_blobs, FileName):

    result = []
    
    for group in cluster_blobs:
        result.append(sorted(group))

    result = sorted(result)

    with open(FileName,"w") as f:

      for blob in result:
        f.write(str(blob)[1:-1])
        f.write("\n")

      f.close()

# ---------------------------------------------------------------------------------------------------


# Function to Calculate Silhouette coefficient ------------------------------------------------------

def calc_silhouette_coeff(X, cluster_blobs, clusters):

    scores = []
    for i in range(0, len(X)):
        cluster_no_of_Xi = clusters[i]
        B = {}
        a = 0.0
        si = 0.0
        sum_intra_clust_dist = 0.0
        bag = []
        Ni = len(cluster_blobs[cluster_no_of_Xi])

        for j in range(len(X)):

            # skip
            if j == i:
                continue

            # Calculating intra-cluster distance
            if clusters[j] == cluster_no_of_Xi:
                sum_intra_clust_dist += euclidean_dist(X[i], X[j])

            # Calculating inter-cluster distance
            else:
                key = clusters[j]

                # If the cluster is already there
                if key in B:
                    B[key] += euclidean_dist(X[i], X[j])
                # If the cluster is not already there
                else:
                    B[key] = euclidean_dist(X[i], X[j])

        # Get average value of B[i] for each cluster
        for key, d in B.items():

            n_key = len(cluster_blobs[key])
            d_avg = (d/n_key)
            bag.append(d_avg)

        # The mean distance between Xi and all other points in the same cluster
        ai  = (sum_intra_clust_dist/ (Ni-1) )

        # Get the distance of the cluster nearest to Xi
        bi = min(bag)

        # Get si
        si = 0.0
        if Ni>1 : 
            si = (bi - ai)/max(ai, bi)

        scores.append(si)

    mean_silhouette_coeff = np.mean(scores, dtype=np.float64)

    return mean_silhouette_coeff

# ---------------------------------------------------------------------------------------------------


# Functions and class required in Heirarchal Clustering ---------------------------------------------

class Cluster:
    
	def __init__(self, id, vecs = None):     

		self.id = id	

		if vecs is None:
			self.vecs = [self.id]
		else:
			self.vecs = vecs[:]


def min_dist_btw_two_clusters(cluster1, cluster2):

    min_dist = 123123123.0

    for i in cluster1:
        for j in cluster2:
            dist_between_Xi_and_Xj = euclidean_dist(X[i], X[j])

            if dist_between_Xi_and_Xj < min_dist:
                min_dist = dist_between_Xi_and_Xj

    return min_dist


def hierarchical_clustering(optimal_k):

    # distances is the cache of distance calculations
	distances = {}
	current_clust_id = -1

	# Individual data points as separate clusters
	cluster_blobs = [Cluster(id=i) for i in range(len(X))]

	while len(cluster_blobs) > optimal_k:

		closest_pair = (0,1)
		closest_dist = 123123123.0
	
		# loop through every pair looking for the smallest distance
		for i in range(len(cluster_blobs)-1):
			for j in range(i+1,len(cluster_blobs)):
       
				# If not already precomputed, calculate the distance
				if (cluster_blobs[i].id,cluster_blobs[j].id) not in distances:
						distances[(cluster_blobs[i].id,cluster_blobs[j].id)] = min_dist_btw_two_clusters(cluster_blobs[i].vecs, cluster_blobs[j].vecs)
		
				D = distances[(cluster_blobs[i].id,cluster_blobs[j].id)]
		
				if D < closest_dist:
					closest_dist = D
					closest_pair = (i,j)
		
		# create the new cluster by merging closest clusters
		new_cluster = Cluster(current_clust_id, vecs = cluster_blobs[closest_pair[0]].vecs + cluster_blobs[closest_pair[1]].vecs)

		# cluster ids that weren't in the original set are negative
		current_clust_id -= 1

        # Delete the two clusters merged
		del cluster_blobs[closest_pair[1]]
		del cluster_blobs[closest_pair[0]]

        # add the new cluster
		cluster_blobs.append(new_cluster)

	return cluster_blobs

# ---------------------------------------------------------------------------------------------------


# main 
if __name__ == '__main__':

    # Read data
    data_set = pd.read_csv('COVID_2_unlabelled.csv')

    X = data_set.iloc[:,1:]
    X = np.array(X)

    # z-score normalization
    X = (X-(np.sum(X,axis=0)/len(X))) / np.std(X,axis=0)


    FileNames = ["clustering_info_3.txt", "clustering_info_4.txt", "clustering_info_5.txt", "clustering_info_6.txt"]
    clustering_info = []
    max_silhouette_coeff = -2.0
    optimal_k = 1

    print("No. of clusters        Silhouette coefficient")
    for k in range(3, 7):

        # Step - 1
        clustering_info_k = k_means(X, k)
        clustering_info.append(clustering_info_k)

        save_clustering_info(clustering_info_k[2], FileNames[k-3])

        # Step - 2
        silhouette_coeff_k = calc_silhouette_coeff(X, clustering_info_k[2], clustering_info_k[3])
        print("     ", k, "         ->    ", silhouette_coeff_k)

        if(silhouette_coeff_k > max_silhouette_coeff):
            max_silhouette_coeff = silhouette_coeff_k
            optimal_k = k

    # Step - 3
    print("Optimal K -> ", optimal_k, '\n')


    # Step -4

    RESULT = hierarchical_clustering(optimal_k)
    hierarchical_clustering_info=[]
    for clust in RESULT:
        hierarchical_clustering_info.append(clust.vecs)


    max_jaccard_index = [0.0 for _ in range(optimal_k)]
    group_with_max_jacc_idx = {} 

    for i in range(optimal_k):

        for j in range(optimal_k):

            groupA = clustering_info[optimal_k-3][2][i]
            groupB = hierarchical_clustering_info[j]

            intersection = len(list(set(groupA).intersection(groupB)))
            union = (len(groupA) + len(groupB)) - intersection

            jaccard_index = float(intersection) / union

            if(jaccard_index > max_jaccard_index[i]):
                max_jaccard_index[i] = jaccard_index
                group_with_max_jacc_idx[i] = j
            
    # print Mappings
    print("Mappings : ", group_with_max_jacc_idx, '\n')

    # print the Jaccard Similarity scores for all the k mappings
    print("Mapping          Jaccard Similarity score")
    for i in range(optimal_k):
        print(i,':',group_with_max_jacc_idx[i],  "     ->          ", max_jaccard_index[i])