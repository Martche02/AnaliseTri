import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
def find_pairs_with_difference(df, X):
    # Ordenar o DataFrame pelos números seriais
    sorted_df = df.sort_values(by='Acertos_Decimal')

    # Converter para array NumPy para acesso mais rápido
    serial_numbers = sorted_df['Acertos_Decimal'].to_numpy()
    notes = sorted_df['NU_NOTA_CN'].to_numpy()
    
    pairs = []
    for i in range(len(serial_numbers)):
        # Busca binária por um par
        target = serial_numbers[i] + X
        j = np.searchsorted(serial_numbers, target, side='left')
        if j < len(serial_numbers) and serial_numbers[j] == target:
            pairs.append((notes[i], notes[j]))
    
    return pairs

def carQ(df:pd.DataFrame, X:int, p:bool=False)->float:
    # df = pd.read_csv("2022dados/1087.0.csv")
    X = 2**X #np.random.randint(1,45)  # A diferença específica que você está procurando



    # Exemplo de uso
    t = time()
    l = set(find_pairs_with_difference(df, X))
    # print(l)
    # print(np.log2(X), X)
    # print(time()-t)
    x_vals = np.array([])
    y_vals = np.array([])

    l = [pair for pair in l if pair[0] != 0 and pair[1]>1]
    for pair in l:
        lower, higher = min(pair), max(pair)
        # if lower>100:
        x_vals = np.append(x_vals,[lower])
        y_vals = np.append(y_vals,[higher-lower])
    # plt.figure(figsize=(10, 6))
    # plt.scatter(x_vals, y_vals, color='blue')
    # plt.title('Scatter Plot of Data Points')
    # plt.xlabel('X values')
    # plt.ylabel('Y values')
    # plt.grid(True)
    # # plt.show()

    # lin_reg = LinearRegression()
    # lin_reg.fit(x_vals.reshape(-1, 1), y_vals)
    # slope = lin_reg.coef_[0]
    # intercept = lin_reg.intercept_

    # # Transformation to flatten the trend line
    # # 1. Translate the data
    # translated_y_vals = y_vals - intercept

    # # 2. Rotate the data
    # # For a line y = mx, the angle of rotation needed is -arctan(m)
    # angle = np.arctan(slope)
    # rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], 
    #                             [np.sin(angle), np.cos(angle)]])
    # transformed_data = np.dot(rotation_matrix, np.vstack((x_vals, translated_y_vals)))

    # # Plot the transformed data
    # plt.figure(figsize=(10, 6))
    # plt.scatter(transformed_data[0, :], transformed_data[1, :], color='blue')
    # plt.title('Transformed Data with Flattened Trend')
    # plt.xlabel('Transformed X values')
    # plt.ylabel('Transformed Y values')
    # plt.grid(True)
    # # np.set_printoptions(threshold=np.inf)
    # # np.savetxt("data.txt", transformed_data[::-1])
    # # print(transformed_data[::-1])
    # plt.show()
    # db = DBSCAN(eps=0.8, min_samples=40)  # Adjust the eps and min_samples parameters as needed
    # db.fit(np.column_stack((x_vals, y_vals)))

    # # Labels of the clusters
    # labels = db.labels_

    # # Number of clusters in labels, ignoring noise if present
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    # # Plot the clusters
    # plt.figure(figsize=(10, 6))
    # unique_labels = set(labels)
    # for k in unique_labels:
    #     if k == -1:
    #         # Black used for noise
    #         col = 'k'
    #     else:
    #         col = plt.cm.Spectral(float(k) / n_clusters_)
    #     class_member_mask = (labels == k)
    #     xy = np.column_stack((x_vals, y_vals))[class_member_mask]
    #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

    # plt.title('DBSCAN Clustering')
    # plt.xlabel('X values')
    # plt.ylabel('Y values')
    # plt.show()
    # kmeans = KMeans(n_clusters=1, random_state=0).fit(np.column_stack((x_vals, y_vals)))
    # labels = kmeans.labels_
    # cluster_1 = np.column_stack((x_vals[labels == 0], y_vals[labels == 0]))
    # cluster_2 = np.column_stack((x_vals[labels == 1], y_vals[labels == 1]))
    # n_clusters = 1  
    # kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(np.column_stack((x_vals, y_vals)))
    # labels = kmeans.labels_
    # def fit_and_plot_with_formula(cluster, title, color, ax):
    #     # Linear regression
    #     lin_reg = LinearRegression()
    #     lin_reg.fit(cluster[:, 0].reshape(-1, 1), cluster[:, 1])
    #     lin_coef = lin_reg.coef_[0]
    #     lin_intercept = lin_reg.intercept_
    #     lin_formula = f'y = {lin_coef:.2f}x + {lin_intercept:.2f}'

    #     lin_x = np.linspace(cluster[:, 0].min(), cluster[:, 0].max(), 100)
    #     lin_y = lin_reg.predict(lin_x.reshape(-1, 1))

    #     # Polynomial regression (degree 2)
    #     poly_reg = make_pipeline(PolynomialFeatures(2), LinearRegression())
    #     poly_reg.fit(cluster[:, 0].reshape(-1, 1), cluster[:, 1])
    #     poly_coef = poly_reg.named_steps['linearregression'].coef_
    #     poly_intercept = poly_reg.named_steps['linearregression'].intercept_
    #     poly_formula = f'y = {poly_coef[2]:.2f}x² + {poly_coef[1]:.2f}x + {poly_intercept:.2f}'

    #     poly_y = poly_reg.predict(lin_x.reshape(-1, 1))

    #     # Plotting
    #     ax.scatter(cluster[:, 0], cluster[:, 1], color=color)
    #     ax.plot(lin_x, lin_y, label='Linear Fit: ' + lin_formula, color='black')
    #     ax.plot(lin_x, poly_y, label='Polynomial Fit (Degree 2): ' + poly_formula, color='red')
    #     ax.set_title(title)
    #     ax.set_xlabel('X values')
    #     ax.set_ylabel('Y values')
    #     ax.legend()

    # # Plotting for each cluster with formulas
    # fig, axs = plt.subplots(1, n_clusters, figsize=(20, 8))

    # if n_clusters == 1:
    #     fit_and_plot_with_formula(np.column_stack((x_vals, y_vals)), 'Curve Fitting for All Data', 'blue', axs)
    # else:
    #     for i in range(n_clusters):
    #         fit_and_plot_with_formula(np.column_stack((x_vals[labels == i], y_vals[labels == i])), 
    #                                   f'Curve Fitting for Cluster {i+1}', 
    #                                   'blue' if i == 0 else 'green', 
    #                                   axs[i] if n_clusters > 1 else axs)

    # plt.show()
    # ... [Import statements and DataFrame preparation]

    # Your data preparation and plotting code remains the same

    # Set the number of clusters to 2
    # n_clusters = 1

    # # Applying K-means clustering
    # kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(np.column_stack((x_vals, y_vals)))
    # labels = kmeans.labels_
    
    # Function to fit and plot with linear regression for each cluster
    def get_fitline(points):
        lin_reg = LinearRegression()
        lin_reg.fit(points[:, 0].reshape(-1, 1), points[:, 1])
        lin_coef = lin_reg.coef_[0]
        lin_intercept = lin_reg.intercept_
        return float(f'{lin_coef:.2f}'),float(f'{lin_intercept:.2f}')
    def fit_and_plot_linear(cluster, title, color, ax):
        # Linear regression
        lin_reg = LinearRegression()
        lin_reg.fit(cluster[:, 0].reshape(-1, 1), cluster[:, 1])

        lin_coef = lin_reg.coef_[0]
        lin_intercept = lin_reg.intercept_
        a,b=f'{lin_coef:.2f}',f'{lin_intercept:.2f}'
        if p:
            lin_formula = f'y = {lin_coef:.2f}x + {lin_intercept:.2f}'

            lin_x = np.linspace(cluster[:, 0].min(), cluster[:, 0].max(), 100)
            lin_y = lin_reg.predict(lin_x.reshape(-1, 1))

            # Plotting
            ax.scatter(cluster[:, 0], cluster[:, 1], color=color)
            ax.plot(lin_x, lin_y, label='Linear Fit: ' + lin_formula, color='black')
            ax.set_title(title)
            ax.set_xlabel('X values')
            ax.set_ylabel('Y values')
            ax.legend()
        return a,b

    # Plotting for each cluster with linear regression formulas
    fig, axs = plt.subplots(1, 1, figsize=(20, 8))
    # x_vals = np.array([...])  # your x-coordinates
    # y_vals = np.array([...])  # your y-coordinates

    # Fit the linear regression model
    lin_reg = LinearRegression()
    lin_reg.fit(x_vals.reshape(-1, 1), y_vals)

    # Get the slope and intercept of the line
    # slope = lin_reg.coef_[0]
    intercept = lin_reg.intercept_

    # Predict y-values
    predicted_y_vals = lin_reg.predict(x_vals.reshape(-1, 1))

    # Filter out points that are below the regression line
    above_line_mask = y_vals > predicted_y_vals
    x_vals = x_vals[above_line_mask]
    y_vals = y_vals[above_line_mask]
    for i in range(1):
        cluster_data = np.column_stack((x_vals, y_vals))
        a,b = fit_and_plot_linear(cluster_data, 
                            f'Linear Fit for Q {int(np.log2(X)+1)}', 
                            'blue' if i == 0 else 'green', 
                            axs[i] if 1 > 1 else axs)
    plt.show() if p else 0
    return float(a),(float(b)+intercept)/2

### LIXO DO DICIONARIO


    # print(achar_melhor_opcao(415549446))
    # df = pd.read_csv(f"2022dadositens/1087.csv")
    # column_to_shift = "Angular_C"
    # # Shift the column
    # df[column_to_shift] = df[column_to_shift].shift(-1)

    # df.iloc[-1, df.columns.get_loc(column_to_shift)] = df[column_to_shift].iloc[0]
    # df.to_csv("2022dadositens/1087.csv")

    # addLine(2022, 1087)
    
    # carQ(pd.read_csv(f"2022dados/1087.0.csv"), 44, True)
    # pd.concat([pd.read_csv("2022dadositens/108586.csv"),pd.read_csv("2022dadositens/1087.csv")], ignore_index=True).to_csv("2022dadositens/10858687.csv")
    # pd.concat([pd.read_csv("2022dados/108586.0.csv"),pd.read_csv("2022dados/1087.0.csv")], ignore_index=True).to_csv("2022dados/10858687.0.csv")
    # addLine(2022,108586)
    # addLine(2022,1087)
    # print(notaProx(118364175,2022,1087))
    # Apply the aproxNota function to the 'Acertos_Decimal' column
    
    
    # import matplotlib.pyplot as plt
    # start = 1
    # end = 45

    # # Generate 400 equally spaced values in the range
    # x_values = np.linspace(start, end)

    # # Calculate the output for each value
    # y_values = [aproxNota(int((45-int(x))*'0'+int(x)*'1',2),2022,1087) for x in x_values]
    # y2_values = [aproxNota(int((int(x))*'1'+(45-int(x))*'0',2),2022,1087) for x in x_values]

    # # Plotting
    # plt.figure(figsize=(12, 6))
    # plt.plot(x_values, y_values, y2_values, label='aproxNota(x)')
    # plt.xlabel('Input Value')
    # plt.ylabel('Output Value')
    # plt.title('Function Behavior of aproxNota')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
