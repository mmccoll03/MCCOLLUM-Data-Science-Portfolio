import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.special

def show_perceptron_training_example():
    st.write("""
    In this example, a logistic regression model is trained using a perceptron-like update rule with gradient descent on randomly generated data.
    
    **Parameters Used:**
    - **Data Points (m):** 100  
    - **Dimensions (n):** 2  
    - **Initial Weights:** w = [4, 1]  
    - **Initial Bias:** b = -20 
    - **Learning Rate:** 0.001  
    - **Iterations:** 2000  
    """)
    
    # --- Data Generation ---
    m = 100
    n = 2
    Xtrain = np.append(4.0 + np.random.randn(m//2, n),
                        6.0 + np.random.randn(m//2, n), axis=0)
    Ytrain = np.append(np.zeros(m//2), np.ones(m//2))
    temp = np.random.permutation(m)
    Xtrain = Xtrain[temp, :]
    Ytrain = Ytrain[temp]
    
    Xtest = np.append(4.0 + np.random.randn(m//2, n),
                       6.0 + np.random.randn(m//2, n), axis=0)
    Ytest = np.append(np.zeros(m//2), np.ones(m//2))
    temp = np.random.permutation(m)
    Xtest = Xtest[temp, :]
    Ytest = Ytest[temp]
    
    # Display the initial scattered data
    st.subheader("Initial Data Visualization")
    fig_init, axs_init = plt.subplots(1, 2, figsize=(12, 4))
    axs_init[0].scatter(Xtrain[Ytrain == 0, 0], Xtrain[Ytrain == 0, 1],
                        color='blue', label="y = 0", alpha=0.7)
    axs_init[0].scatter(Xtrain[Ytrain == 1, 0], Xtrain[Ytrain == 1, 1],
                        color='red', label="y = 1", alpha=0.7)
    axs_init[0].set_xlabel("x1")
    axs_init[0].set_ylabel("x2")
    axs_init[0].set_title("Training Data")
    axs_init[0].legend()
    
    axs_init[1].scatter(Xtest[Ytest == 0, 0], Xtest[Ytest == 0, 1],
                        color='blue', label="y = 0", alpha=0.7)
    axs_init[1].scatter(Xtest[Ytest == 1, 0], Xtest[Ytest == 1, 1],
                        color='red', label="y = 1", alpha=0.7)
    axs_init[1].set_xlabel("x1")
    axs_init[1].set_ylabel("x2")
    axs_init[1].set_title("Test Data")
    axs_init[1].legend()
    st.pyplot(fig_init)
    
    # --- Train the Model on Button Click ---
    if st.button("Train Perceptron Model"):
        # Define helper functions
        def sigma(z):
            return np.where(z > 0, 1/(1 + np.exp(-z)), np.exp(z)/(1 + np.exp(z)))
        def g(z):
            return sigma(z)
        def Loss(Yhat, Y):
            L = np.zeros_like(Y)
            L[Y == 0] = scipy.special.logsumexp([np.zeros_like(Yhat[Y == 0]), Yhat[Y == 0]], axis=0)
            L[Y == 1] = scipy.special.logsumexp([np.zeros_like(Yhat[Y == 1]), -Yhat[Y == 1]], axis=0)
            return L
        def GetAccuracy(Yhat, Y):
            Guesses = np.heaviside(Yhat - 0.5, 1)
            return 1 - np.mean(np.abs(Guesses - Y))
        
        # ---- Define f function so that it is available for plotting ----
        def f(x, w, b):
            # Computes the logistic output for input x, weights w, and bias b
            z = x.dot(w) + b
            return g(z)
        
        # --- Training Parameters ---
        w = np.array([4.0, 1.0])
        b = -20
        niterations = 2000
        learning_rate = 0.001
        ComputeTestLoss = True
        
        TrainingLosses = np.zeros(niterations + 1)
        TrainingAccuracies = np.zeros(niterations + 1)
        if ComputeTestLoss:
            TestLosses = np.zeros(niterations + 1)
            TestAccuracies = np.zeros(niterations + 1)
        
        # --- Training Loop ---
        for i in range(niterations):
            ZhatTrain = w @ Xtrain.T + b
            YhatTrain = g(ZhatTrain)
            TrainingLosses[i] = np.mean(Loss(YhatTrain, Ytrain))
            TrainingAccuracies[i] = GetAccuracy(YhatTrain, Ytrain)
            if ComputeTestLoss:
                ZhatTest = w @ Xtest.T + b
                YhatTest = g(ZhatTest)
                TestLosses[i] = np.mean(Loss(YhatTest, Ytest))
                TestAccuracies[i] = GetAccuracy(YhatTest, Ytest)
            # Gradient descent update
            m_train = Xtrain.shape[0]
            Deltaw = -(learning_rate / m_train) * (Xtrain.T @ (YhatTrain - Ytrain))
            Deltab = -(learning_rate / m_train) * np.sum(YhatTrain - Ytrain)
            w = w + Deltaw
            b = b + Deltab
        
        # Final metrics calculation
        ZhatTrain = w @ Xtrain.T + b
        YhatTrain = g(ZhatTrain)
        TrainingLosses[niterations] = np.mean(Loss(YhatTrain, Ytrain))
        TrainingAccuracies[niterations] = GetAccuracy(YhatTrain, Ytrain)
        if ComputeTestLoss:
            ZhatTest = w @ Xtest.T + b
            YhatTest = g(ZhatTest)
            TestLosses[niterations] = np.mean(Loss(YhatTest, Ytest))
            TestAccuracies[niterations] = GetAccuracy(YhatTest, Ytest)
        
        # --- Display Final Parameters and Metrics ---
        st.subheader("Final Model Parameters and Metrics")
        st.write(f"**Final weights (w):** {w}")
        st.write(f"**Final bias (b):** {b}")
        st.write(f"**Learning Rate:** {learning_rate}")
        st.write(f"**Iterations:** {niterations}")
        st.write(f"**Final Training Loss:** {TrainingLosses[-1]:.3f}")
        st.write(f"**Final Training Accuracy:** {TrainingAccuracies[-1]:.3f}")
        if ComputeTestLoss:
            st.write(f"**Final Test Loss:** {TestLosses[-1]:.3f}")
            st.write(f"**Final Test Accuracy:** {TestAccuracies[-1]:.3f}")
        
        # --- Plot Loss and Accuracy Curves ---
        st.subheader("Loss and Accuracy over Iterations")
        fig_curves, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
        ax_loss.plot(TrainingLosses, label="Training Loss")
        if ComputeTestLoss:
            ax_loss.plot(TestLosses, label="Test Loss")
            ax_loss.legend()
        ax_loss.set_xlabel("Iteration Number")
        ax_loss.set_ylabel("Loss")
        ax_acc.plot(TrainingAccuracies, label="Training Accuracy")
        if ComputeTestLoss:
            ax_acc.plot(TestAccuracies, label="Test Accuracy")
            ax_acc.legend()
        ax_acc.set_xlabel("Iteration Number")
        ax_acc.set_ylabel("Accuracy")
        st.pyplot(fig_curves)
        
        # --- Plotting Visualization: Scatter and Histogram on Training Data ---
        plt.subplots(1,2,figsize=(12,4))
        plt.subplot(1,2,1)
        plt.plot(Xtrain[Ytrain==0,0],Xtrain[Ytrain==0,1],'b.')
        plt.plot(Xtrain[Ytrain==1,0],Xtrain[Ytrain==1,1],'r.')
        xplot = np.linspace(1,10,20)
        plt.plot(xplot, (-w[0]/w[1])*xplot - b/w[1], 'm')
        plt.plot(xplot, (w[1]/w[0])*xplot+1, 'k--')
        plt.legend(["y=0","y=1","sep","w"])
        XX1, XX2 = np.meshgrid(np.linspace(1,10,20), np.linspace(1,10,20))
        plt.pcolor(XX1, XX2, np.reshape(f(np.transpose([np.reshape(XX1,-1), np.reshape(XX2,-1)]), w, b), [20,20]), cmap="bwr", alpha=0.2)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.axis([1,10,1,10])
        plt.colorbar()
        plt.title('Training')
        
        plt.subplot(1,2,2)
        wdotx = w @ Xtrain.T
        plt.hist(wdotx[Ytrain==0], 10, color='b', alpha=0.2)
        plt.hist(wdotx[Ytrain==1], 10, color='r', alpha=0.2)
        plt.axvline(x=-b)
        plt.legend(["y=0","y=1",r"$\theta=-b$"])
        plt.xlabel(r'$w\cdot x$')
        plt.ylabel('count')
        plt.title('Training')
        st.pyplot(plt.gcf())

        if ComputeTestLoss:
            plt.subplots(1,2,figsize=(12,4))
            plt.subplot(1,2,1)
            plt.plot(Xtest[Ytest==0,0], Xtest[Ytest==0,1], 'b.')
            plt.plot(Xtest[Ytest==1,0], Xtest[Ytest==1,1], 'r.')
            xplot = np.linspace(1,10,20)
            plt.plot(xplot, (-w[0]/w[1])*xplot - b/w[1], 'm')
            plt.plot(xplot, (w[1]/w[0])*xplot+1, 'k--')
            plt.legend(["y=0","y=1","sep","w"])
            XX1,XX2 = np.meshgrid(np.linspace(1,10,20), np.linspace(1,10,20))
            plt.pcolor(XX1,XX2, np.reshape(f(np.transpose([np.reshape(XX1,-1), np.reshape(XX2,-1)]), w, b), [20,20]), cmap="bwr", alpha=0.2)
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.axis([1,10,1,10])
            plt.colorbar()
            plt.title('Test')
            
            plt.subplot(1,2,2)
            wdotx = w @ Xtest.T
            plt.hist(wdotx[Ytest==0], 10, color='b', alpha=0.2)
            plt.hist(wdotx[Ytest==1], 10, color='r', alpha=0.2)
            plt.axvline(x=-b)
            plt.legend(["y=0","y=1",r"$\theta=-b$"])
            plt.xlabel(r'$w\cdot x$')
            plt.ylabel('count')
            plt.title('Test')
            st.pyplot(plt.gcf())



def show():
    st.markdown("""
    <h1 style='text-align: center; font-family: Garamond, serif; color: #f0f0f0;'>
        Logistic Regression on a Perceptron
    </h1>
    """, unsafe_allow_html=True)
    
    st.write("""
             
    Imagine you had a 2-dimensional set of data points that were red and blue—if you wanted to put
    a line through the cloud of points that best divided them by color, how would you do it? That’s 
    the idea behind logistic regression on a perceptron. A perceptron is a basic model that takes inputs
    (like the x and y coordinates of your points), multiplies them by some weights, adds a bias (just a constant),
    and passes the result through a function to decide whether a point is red or blue. Logistic regression adds a twist:
    instead of making a hard yes-or-no decision like a classic perceptron, it uses the sigmoid function, which squashes
    the result into a probability between 0 and 1. So instead of saying “definitely red” or “definitely blue,” it says
    “this is 80% likely to be blue,” which is great when your points aren’t perfectly separable. Then, using math to tweak
    the weights and bias, it learns the best line that splits the red and blue points as accurately as possible.          

    In this interactive example, random two-dimensional data points are
    generated for two classes (0 and 1). The model uses a perceptron-style
    gradient descent to update its parameters (weights and bias) iteratively.
    By clicking the **"Train Perceptron Model"** button, the training process starts.
    You'll then see the loss and accuracy plotted over iterations, a scatterplot of
    the training data with the decision boundary, and histograms of the projected inputs.
    Finally, the learned logistic regression equation is displayed in LaTeX format,
    showcasing the fitted parameters.
             
    Note that you could tune parameters like the learning rate, weight parameters, and bias, but I've found that
    these parameters do a good job in most cases. 
    """)
    
    # Call the interactive training example function
    show_perceptron_training_example()
