
import matplotlib.animation as animation
import matplotlib
import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt


class scan_trj:
    
    points = ['CAZ','CAY','CBL','CAV','CBK',]
    num_points_per_PAC = len(points)
    times = 1
    num_DRG = 24
    dim_trjs = 3
    num_vecs = 2
    

    def __init__(self,file_dir,crit_dist,crit_cos) -> None:
        self.crit_cos = crit_cos
        self.crit_dist = crit_dist
        with open(file_dir, 'r') as file:
            data = file.readlines()
            num_mol = int(data[1])
            
            self.times = len(data) // (num_mol + 3)
            self.dims = np.zeros(self.times)
            self.trjs = np.zeros((self.times,self.num_DRG,self.num_points_per_PAC,self.dim_trjs))
            self.plns = np.zeros((self.times,self.num_DRG,self.num_vecs,self.dim_trjs))
            self.surfaces = [[] for _ in range(self.times)]
            self.posits = np.zeros((self.times,self.num_DRG,2,self.dim_trjs))
            self.adj = np.zeros((self.times,self.num_DRG,self.num_DRG))
            self.adj_av = np.zeros((self.num_DRG,self.num_DRG))
            self.angs = np.zeros((self.times,self.num_DRG,self.num_DRG))
            self.angs_av = np.zeros((self.num_DRG,self.num_DRG))
            self.dist = np.zeros((self.times,self.num_DRG,self.num_DRG))
            self.dist_av = np.zeros((self.num_DRG,self.num_DRG))
            self.clusters = [[] for _ in range(self.times)]
            self.clusters_av = []
            for time in range(self.times):
                self.dims[time] = float(data[(time+1)* (num_mol+3)-1][:9])
                for dat in data[ time* (num_mol+3) + 2:(time+1)* (num_mol+3)-1]:
                    if dat[5:8] == 'DRG' and dat[12:15] in self.points:
                        self.trjs[time,int(dat[:5])-1,self.points.index(dat[12:15]),:] = np.array([float(dat[20:28]),float(dat[28:36]),float(dat[36:])])
        self.planes()
        self.get_angist()
        self.identify_clusters()
        
#         self.draw_scene()
    
    
    def planes(self):
        pca_plns = PCA(n_components=self.num_vecs)
        width = 0.5
        length = 0.5
        for t in range(self.times):
            for drg in range(self.num_DRG):
                pca_plns.fit(self.trjs[t,drg,:,:])
                self.plns[t,drg,:,:] = pca_plns.components_
                cent = np.average(self.trjs[t,drg,:,:],axis = 0)
                self.posits[t,drg,0,:] = cent
                self.posits[t,drg,1,:] = np.cross(pca_plns.components_[0], pca_plns.components_[1])
                cent = np.matmul(np.ones((4,1)),cent.reshape((1,-1)))
                base1 = np.matmul(np.array([-width,width,width,-width]).reshape((4,1)),pca_plns.components_[0].reshape((1,-1)))
                base2 = np.matmul(np.array([-length,-length,length,length]).reshape((4,1)),pca_plns.components_[1].reshape((1,-1)))
                sur_temp = []
                temp = cent + base1 + base2
                for trj in list(temp):
                    sur_temp.append(tuple(trj))
                self.surfaces[t].append([sur_temp])
    
    def draw_scene(self):
        colors = ['red','blue','green','orange','cyan','black','darkgrey','sienna','lime','purple']
        clrs = {}
        def update_scats(time):
            for drg in range(self.num_DRG):
                ax.scatter(self.trjs[time,drg,:,0], self.trjs[time,drg,:,1], self.trjs[time,drg,:,2],c=clrs[drg])
        
        for t in [0,-1]:
            fig = plt.figure()
            ax = Axes3D(fig,auto_add_to_figure=False)
            fig.add_axes(ax)
            for j,clst in enumerate(self.clusters[t]):
                for node in clst:
                    clrs[node] = colors[j%len(colors)]
            for drg in range(self.num_DRG):
                ax.scatter(self.trjs[t,drg,:,0], self.trjs[t,drg,:,1], self.trjs[t,drg,:,2],c=clrs[drg])
            for i ,surface in enumerate(self.surfaces[t]):
                ax.add_collection3d(Poly3DCollection(surface,facecolors=clrs[i],alpha=.50))
        # ani = animation.FuncAnimation(fig, update_scats, frames=self.times)
        
        #     for drg in range(self.num_DRG):
        #         ax.scatter(self.trjs[t,drg,:,0], self.trjs[t,drg,:,1], self.trjs[t,drg,:,2],c=clrs[drg])
        #     for i ,surface in enumerate(self.surfaces[t]):
        #         ax.add_collection3d(Poly3DCollection(surface,facecolors=clrs[i],alpha=.50))
        # ani.save('matplot003.gif', writer='imagemagick')
        
            # plt.savefig(f'figs/time{t}.png', dpi='figure')
            # plt.clf()
    
    def get_angist(self):
        for t in range(self.times):
            self.angs[t,:,:] = np.abs(self.posits[t,:,1,:].dot(self.posits[t,:,1,:].transpose()))
            norms = np.linalg.norm(self.posits[t,:,1,:],axis=1,keepdims=True)
            self.angs[t,:,:] /= np.matmul(norms,norms.transpose())
            # print(self.angs)
            d1 = np.tile(self.posits[t,:,0,:].reshape(1,self.num_DRG,3), (self.num_DRG, 1, 1))
            d2 = np.tile(self.posits[t,:,0,:].reshape(self.num_DRG,1,3), (1,self.num_DRG, 1))
            self.dist[t,:,:] = np.linalg.norm(d1-d2,axis=2)
            self.dist[t,:,:] = np.minimum(self.dist[t,:,:],self.dims[t]-self.dist[t,:,:])
            self.adj[t,:,:] = np.logical_and(self.angs[t,:,:] > self.crit_cos , self.dist[t,:,:] < self.crit_dist)
            # print(self.adj[t,:,:])
        self.angs_av = np.average(self.angs,axis=0)
        self.dist_av = np.average(self.dist,axis=0)
        self.adj_av = np.logical_and(self.angs_av > self.crit_cos , self.dist_av < self.crit_dist)
    
    def identify_clusters(self):
        for t in range(self.times):
            visited = [False for _ in range(self.num_DRG)]
            
            while False in visited:
                node = visited.index(False)
                temp_list = []
                self.visit(node,visited,temp_list,self.adj[t,:,:])
                # print(temp_list)
                self.clusters[t].append(temp_list)
            # print(self.clusters[t])
        visited = [False for _ in range(self.num_DRG)]
            
        while False in visited:
            node = visited.index(False)
            temp_list = []
            self.visit(node,visited,temp_list,self.adj_av)
            # print(temp_list)
            self.clusters_av.append(temp_list)
        # print(self.clusters[t])
        
        
    def visit(self, node , vis_list, comp_list,adj_mat):
        if vis_list[node] :
            return
        vis_list[node] = True
        comp_list.append(node)
        for i,neigh in enumerate(adj_mat[node,:]):
            if neigh:
                self.visit(i,vis_list,comp_list,adj_mat)
    
    def binize(self,num_bin):
        min_bin,max_bin = 0 , np.max(self.dist_av)
        bins = np.linspace(min_bin,max_bin,num_bin) 
#         for t in [0,-1]:#range(self.times):
        dis_bin = np.digitize(self.dist_av,bins)
        angbin = np.zeros(num_bin)
        for j in range(num_bin):
            ind = np.where(dis_bin==j+1)
#             angs_temp = self.angs[t,:,:]
            angbin[j] = np.average(self.angs_av[ind])
        plt.plot(bins,angbin,marker='o',color = 'red')
         
    
    def plt_all(self):
        ind = np.triu_indices(self.num_DRG,k=1)
#         for t in range(self.times):
            
#             dist_t = self.dist[t,:,:]
#             angs_t = self.angs[t,:,:]
            
        plt.scatter(self.dist_av[ind],self.angs_av[ind])

    def cluster_angist(self):
        import itertools
        colors = ['red','blue','green','orange','cyan','black','darkgrey','sienna','lime','purple']
#         for t in range(self.times):
#             dist_t = self.dist[t,:,:]
#             angs_t = self.angs[t,:,:]
        for c,clust in enumerate(self.clusters_av):
            ind0 = [z[0] for z in itertools.product(clust, clust)]
            ind1 = [z[1] for z in itertools.product(clust, clust)]
            clust_dist = self.dist_av[ind0,ind1].reshape((len(clust),len(clust)))
            clust_angs = self.angs_av[ind0,ind1].reshape((len(clust),len(clust)))
            ind_ = np.triu_indices(len(clust),k=1)
#                 print(clust,ind0,ind_)
            plt.scatter(clust_dist[ind_],clust_angs[ind_],color=colors[c%len(colors)])


n_cor = 16
n_samp = 5
n_dim = 3
data_test = np.zeros((n_samp,n_cor,n_dim))
r = np.linspace(0.0,1,n_cor).reshape((-1,1))

n_cor_sq = int(n_cor**0.5)
rc = np.linspace(0.0,1,n_cor_sq)
list_of_comps = []
pca_test = PCA(n_components=2)
r2 = np.array([0,0,1])
wid = 0.5
hei = 0.5
surfs = []
for i in range(n_samp):
    r1 = np.array([np.cos(np.pi*(i/n_samp)),np.sin(np.pi*(i/n_samp)),0])
    
    # plt.scatter(r1[0],r1[1])
    for j in range(n_cor_sq):
        for k in range(n_cor_sq):
            data_test[i , j*n_cor_sq + k ,:] = r1*rc[j] + r2*rc[k] +  np.random.randn(n_dim)*0.03
    
    # ax.scatter(data_test[i,:,0], data_test[i,:,1], data_test[i,:,2],c=colors[i])
    pca_test.fit(data_test[i,:,:])
    # print(pca_test.components_)
    comp1 ,comp2 = pca_test.components_
    cent = np.average(data_test[i,:,:],axis = 0).reshape((-1,1))
    # print(cent,comp1,comp2)
    comp1 = comp1.reshape((-1,1))
    comp2 = comp2.reshape((-1,1))
    temp = np.concatenate((cent-comp1*wid-comp2*hei,cent+comp1*wid-comp2*hei,cent+comp1*wid+comp2*hei,cent-comp1*wid+comp2*hei), axis=1)
    # print(temp)
    
    sur_temp = []
    for trj in list(temp.transpose()):
        sur_temp.append(tuple(trj))
        # print(tuple(trj))

    surfs.append([sur_temp])
    
    # print(f"for {i} ({r1}), comps are {temp1} and {temp2}")

# for i ,surface in enumerate(surfs):
    
    # ax.add_collection3d(Poly3DCollection(surface,facecolors=colors[i],alpha=.20))

# plt.show()


# tr_test = scan_trj('snaps/snap.gro')
# plt.clf()
# print(tr_test.dims)
# plt.show()
