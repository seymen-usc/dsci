# ── Install / upgrade ipywidgets for Google Colab ──
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ipywidgets'], check=True)
print('✅ Dependencies ready!')

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


