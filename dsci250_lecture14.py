# ── Install / upgrade ipywidgets for Google Colab ──
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ipywidgets'], check=True)
print('✅ Dependencies ready!')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, mean_squared_error, r2_score
)
from sklearn.decomposition import PCA

import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, HBox, VBox, Output, Layout
from IPython.display import display, clear_output

# Consistent random seed
np.random.seed(42)

# Plot style
plt.rcParams.update({'figure.dpi': 100, 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})

def kmeans():
    np.random.seed(1)
    X_km, _ = make_blobs(n_samples=250, centers=4, cluster_std=1.1, random_state=1)
    
    out_km = Output()
    
    def plot_kmeans(K=4, show_elbow=False, n_init=10):
        with out_km:
            clear_output(wait=True)
            km = KMeans(n_clusters=K, n_init=n_init, random_state=42)
            labels = km.fit_predict(X_km)
            centers = km.cluster_centers_
            inertia = km.inertia_
    
            cols = 2 if show_elbow else 1
            fig, axes = plt.subplots(1, cols, figsize=(14 if show_elbow else 6.5, 5))
            ax = axes[0] if show_elbow else axes
    
            palette = ['#E63946','#457B9D','#2A9D8F','#E9C46A','#F4A261','#264653']
            for k in range(K):
                mask = labels == k
                ax.scatter(*X_km[mask].T, color=palette[k % len(palette)],
                           alpha=0.7, edgecolors='k', linewidths=0.3, s=45,
                           label=f'Cluster {k+1}')
            ax.scatter(*centers.T, color='black', marker='X', s=200, zorder=5, label='Centroids')
            ax.set_title(f'K-Means  K={K}  |  Inertia={inertia:.0f}')
            ax.set_xlabel('Study Hours'); ax.set_ylabel('GPA (scaled)')
            ax.legend(fontsize=8)
    
            if show_elbow:
                ks = range(1, 10)
                inertias = [KMeans(n_clusters=k, n_init=10, random_state=42)
                            .fit(X_km).inertia_ for k in ks]
                axes[1].plot(ks, inertias, 'bo-', lw=2)
                axes[1].axvline(x=4, color='crimson', ls='--', label='Elbow ≈ 4')
                axes[1].set_xlabel('Number of Clusters K'); axes[1].set_ylabel('Inertia (WCSS)')
                axes[1].set_title('Elbow Method — pick K at the bend')
                axes[1].legend()
    
            plt.tight_layout(); plt.show()
    
    w_K     = widgets.IntSlider(value=4, min=1, max=16, description='K:', layout=Layout(width='340px'))
    w_elbow = widgets.Checkbox(value=False, description='Show Elbow plot')
    
    display(HBox([w_K, w_elbow]), out_km)
    widgets.interactive_output(plot_kmeans, {'K': w_K, 'show_elbow': w_elbow})
    plot_kmeans()
def knn():
    np.random.seed(3)
    X_knn, y_knn = make_classification(
        n_samples=150, n_features=2, n_redundant=0, n_informative=2,
        n_clusters_per_class=1, class_sep=1.0, random_state=3)
    
    out_knn = Output()
    
    def plot_knn(K=5, metric='euclidean', new_x=0.0, new_y=0.0, show_new=False):
        with out_knn:
            clear_output(wait=True)
            clf = KNeighborsClassifier(n_neighbors=K, metric=metric)
            clf.fit(X_knn, y_knn)
            score = clf.score(X_knn, y_knn)
    
            h = 0.05
            x_min, x_max = X_knn[:,0].min()-0.5, X_knn[:,0].max()+0.5
            y_min, y_max = X_knn[:,1].min()-0.5, X_knn[:,1].max()+0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                  np.arange(y_min, y_max, h))
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
            fig, ax = plt.subplots(figsize=(7, 5))
            cmap_bg  = ListedColormap(['#FFDDDD','#DDDDFF'])
            cmap_pts = ListedColormap(['#CC2222','#2222CC'])
            ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_bg)
            ax.scatter(*X_knn.T, c=y_knn, cmap=cmap_pts, edgecolors='k', s=40, alpha=0.8)
    
            if show_new:
                pt = np.array([[new_x, new_y]])
                pred = clf.predict(pt)[0]
                dists, idxs = clf.kneighbors(pt)
                for idx in idxs[0]:
                    ax.plot([new_x, X_knn[idx,0]], [new_y, X_knn[idx,1]],
                            'k--', lw=1, alpha=0.5)
                ax.scatter(*X_knn[idxs[0]].T, s=120, facecolors='none',
                           edgecolors='gold', linewidths=2.5, zorder=5, label=f'{K} neighbours')
                color = '#CC2222' if pred == 0 else '#2222CC'
                ax.scatter(new_x, new_y, color=color, s=200, marker='*',
                           edgecolors='black', linewidths=1.5, zorder=6,
                           label=f'Prediction: Class {pred}')
                ax.legend(fontsize=9)
    
            ax.set_title(f'KNN  K={K}  metric={metric}  (train acc={score:.2f})')
            ax.set_xlabel('Feature 1 (e.g. Class Success)'); ax.set_ylabel('Feature 2 (e.g. Hours Studied)')
            plt.tight_layout(); plt.show()
    
    w_k       = widgets.IntSlider(value=5, min=1, max=30, description='K:', layout=Layout(width='340px'))
    w_show_pt = widgets.Checkbox(value=False, description='Place test point')
    w_nx      = widgets.FloatSlider(value=0.0, min=-3.0, max=3.0, step=0.1,
                                     description='X:', layout=Layout(width='300px'))
    w_ny      = widgets.FloatSlider(value=0.0, min=-3.0, max=3.0, step=0.1,
                                     description='Y:', layout=Layout(width='300px'))
    
    display(HBox([w_k]),
            HBox([w_show_pt, w_nx, w_ny]),
            out_knn)
    widgets.interactive_output(plot_knn, {'K': w_k,
                                           'new_x': w_nx, 'new_y': w_ny, 'show_new': w_show_pt})
    plot_knn()
def decision_tree():
    np.random.seed(0)
    X_dt, y_dt = make_classification(
        n_samples=200, n_features=2, n_redundant=0, n_informative=2,
        n_clusters_per_class=1, class_sep=0.7, random_state=0)
    X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(
        X_dt, y_dt, test_size=0.3, random_state=42)
    
    out_dt = Output()
    
    def plot_dt(max_depth=2, show_tree=False):
        with out_dt:
            clear_output(wait=True)
            clf = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
            clf.fit(X_train_dt, y_train_dt)
            train_acc = clf.score(X_train_dt, y_train_dt)
            test_acc  = clf.score(X_test_dt,  y_test_dt)
    
            h = 0.05
            x_min, x_max = X_dt[:,0].min()-0.5, X_dt[:,0].max()+0.5
            y_min, y_max = X_dt[:,1].min()-0.5, X_dt[:,1].max()+0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                  np.arange(y_min, y_max, h))
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
            cols = 2 if show_tree else 1
            fig, axes = plt.subplots(1, cols, figsize=(14 if show_tree else 6.5, 4.5))
            ax = axes[0] if show_tree else axes
    
            cmap_bg  = ListedColormap(['#FFCCCC','#CCCCFF'])
            cmap_pts = ListedColormap(['#CC0000','#0000CC'])
            ax.contourf(xx, yy, Z, alpha=0.35, cmap=cmap_bg)
            ax.scatter(*X_train_dt.T, c=y_train_dt, cmap=cmap_pts, edgecolors='k', s=35, alpha=0.8, label='Train')
            ax.scatter(*X_test_dt.T,  c=y_test_dt,  cmap=cmap_pts, edgecolors='white', s=55, marker='D', alpha=0.9, label='Test')
            ax.set_title(f'Max Depth = {max_depth}\nTrain acc: {train_acc:.2f}  |  Test acc: {test_acc:.2f}')
            ax.set_xlabel('Attendance'); ax.set_ylabel('Homework Success')
            ax.legend(loc='upper right', fontsize=9)
    
            if test_acc < train_acc - 0.07:
                ax.set_title(ax.get_title() + '  ⚠️ Overfitting!', color='crimson')
    
            if show_tree:
                plot_tree(clf, ax=axes[1], feature_names=['Attendance','Homework Success'],
                          class_names=['Fail','Pass'], filled=True, rounded=True,
                          fontsize=8, impurity=False)
                axes[1].set_title('Tree Structure')
    
            plt.tight_layout(); plt.show()
    
    w_depth    = widgets.IntSlider(value=2, min=1, max=12, step=1,
                                    description='Max Depth:', style={'description_width':'80px'},
                                    layout=Layout(width='380px'))
    w_showtree = widgets.Checkbox(value=False, description='Show tree structure')
    
    display(HBox([w_depth, w_showtree]), out_dt)
    widgets.interactive_output(plot_dt, {'max_depth': w_depth, 'show_tree': w_showtree})
    plot_dt(2, False)
def linear_regression():

  # Generate data: exam score vs hours studied (with a burnout curve)
  np.random.seed(7)
  hours = np.linspace(0, 12, 10)
  score = 50 + 7*hours - 0.5*hours**2 + np.random.normal(0, 5, 10)
  score = np.clip(score, 0, 100)
  
  # Precompute the actual best-fit line (for optional reference/comparison)
  X = hours.reshape(-1, 1)
  best_model = LinearRegression().fit(X, score)
  best_intercept = best_model.intercept_
  best_slope = best_model.coef_[0]
  sst = np.sum((score - score.mean())**2)
  
  out_lin = Output()
  
  def plot_linear(intercept=50.0, slope=0.0, show_residuals=False, show_best_fit=False):
      with out_lin:
          clear_output(wait=True)
          x_plot = np.linspace(0, 12, 300)
          y_pred_plot = intercept + slope * x_plot
          y_pred_data = intercept + slope * hours
  
          residuals = score - y_pred_data
          sse = np.sum(residuals**2)
          mse = sse / len(score)
          r2 = 1 - sse / sst
  
          fig, axes = plt.subplots(1, 2 if show_residuals else 1,
                                    figsize=(13 if show_residuals else 7, 4.5))
          ax = axes[0] if show_residuals else axes
  
          ax.scatter(hours, score, color='steelblue', alpha=0.6, label='Student data', zorder=3)
          ax.plot(x_plot, y_pred_plot, color='crimson', lw=2.5,
                  label=f'Your fit: y = {intercept:.1f} + {slope:.2f}x')
  
          if show_best_fit:
              y_best_plot = best_intercept + best_slope * x_plot
              ax.plot(x_plot, y_best_plot, color='gray', lw=2, ls='--', alpha=0.7,
                       label=f'Best fit: y = {best_intercept:.1f} + {best_slope:.2f}x')
  
          # draw residual lines on the main plot for visual feedback
          for xh, yh, yp in zip(hours, score, y_pred_data):
              ax.plot([xh, xh], [yh, yp], color='darkorange', lw=1, alpha=0.5)
  
          ax.set_xlabel('Hours Studied'); ax.set_ylabel('Exam Score')
          ax.set_title('Manual Linear Fit — adjust intercept & slope')
          ax.legend(loc='upper right'); ax.set_ylim(0, 110)
          ax.set_xlabel(f'Hours Studied\nSSE = {sse:.1f}   MSE = {mse:.2f}   R² = {r2:.3f}',
                        fontsize=10)
  
          if show_residuals:
              axes[1].axhline(0, color='gray', lw=1, ls='--')
              axes[1].scatter(hours, residuals, color='darkorange', alpha=0.7)
              axes[1].set_xlabel('Hours Studied'); axes[1].set_ylabel('Residual')
              axes[1].set_title('Residuals (should be random & centred at 0)')
              axes[1].set_ylim(-30, 30)
  
          plt.tight_layout(); plt.show()
  
  w_intercept = widgets.FloatSlider(value=50.0, min=0, max=100, step=1,
                                     description='Intercept:', style={'description_width':'60px'},
                                     layout=Layout(width='380px'))
  w_slope = widgets.FloatSlider(value=0.0, min=-5, max=15, step=0.1,
                                 description='Slope:', style={'description_width':'60px'},
                                 layout=Layout(width='380px'))
  w_resid = widgets.Checkbox(value=False, description='Show residual plot')
  w_best  = widgets.Checkbox(value=False, description='Show best fit (reference)')
  
  ui = HBox([VBox([w_intercept, w_slope]), VBox([w_resid, w_best])])
  widgets.interactive_output(plot_linear, {
      'intercept': w_intercept, 'slope': w_slope,
      'show_residuals': w_resid, 'show_best_fit': w_best
  })
  
  display(ui, out_lin)
  plot_linear(50.0, 0.0, False, False)



