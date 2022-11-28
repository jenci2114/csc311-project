import matplotlib.pyplot as plt
import ae


def vis_mat_img(raw_mat): 
    plt.imshow(raw_mat)
    plt.savefig(raw_mat)


def vis_latent_hist(latent_mat, latent_as_col):
    
    if latent_as_col:
        latent_dim = latent_mat.shape[0]  # The dimension of one column = # rows
        fig, axes = plt.subplots(nrows=2, ncols=5)
        axes = axes.ravel()
        for i in range(latent_dim):
            axes[i].hist(latent_mat[i, :], bins=20)
            axes[i].set_title(f"latent dim {i}")

        # plt.hist(latent_mat[0, :], bins=20)
        fig.tight_layout()
        plt.savefig("latent_hist.png")
        plt.clf()
        
    else: # latent as row
        latent_dim = latent_mat.shape[1]  # The dimension of one row = # columns
        fig, axes = plt.subplots(nrows=2, ncols=5)
        axes = axes.ravel()
        for i in range(latent_dim):
            axes[i].hist(latent_mat[i, :], bins=30)
            axes[i].set_title(f"latent dim {i}")
        
        plt.savefig("latent_vis.png")
    

def vis_latent_img(latent_mat):
    plt.imshow(latent_mat, cmap='gray')
    



if __name__ == "__main__":
    zero_train_matrix, train_matrix, valid_data, test_data = ae.load_data()
    
    k =10
    lr =0.1
    num_epoch = 10
    model = ae.AutoEncoder(
        num_students=train_matrix.shape[0], 
        k=k, 
        extra_latent_dim=0
        )
    model =ae.train(model=model,
                    lr=lr,
                    lamb=None,
                    train_data=train_matrix,
                    zero_train_data=zero_train_matrix,
                    valid_data=valid_data,
                    num_epoch=num_epoch,
                    betas=None
                    )
    latent_mat = ae.get_latent_mat(
                    model=model,
                    zero_train_data=zero_train_matrix,
                    entity='question'
                    )
    
    vis_latent_hist(latent_mat=latent_mat, latent_as_col=True)
    