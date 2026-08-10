import numpy as np
import os

class LossLandscapeVisualizer:
    """
    Renders loss landscapes in 3D (Plotly, Matplotlib) and 2D contours.
    """
    def __init__(self, x_coords, y_coords, loss_grid):
        """
        Initialize the visualizer.
        
        Args:
            x_coords (array-like): 1D array of coordinate values for the x-axis.
            y_coords (array-like): 1D array of coordinate values for the y-axis.
            loss_grid (np.ndarray): 2D array of loss values corresponding to the grid.
        """
        self.x_coords = np.array(x_coords)
        self.y_coords = np.array(y_coords)
        self.loss_grid = np.array(loss_grid)
        
        # Grid shapes
        self.X, self.Y = np.meshgrid(self.x_coords, self.y_coords)
        
    def _import_matplotlib(self):
        """
        Helper to safely import matplotlib.pyplot, falling back to 'Agg' backend
        if GUI initialization (e.g. Tkinter TclError) fails.
        """
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            return plt
        except Exception:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                return plt
            except ImportError:
                raise ImportError(
                    "Matplotlib is required for static plots. Install it via 'pip install matplotlib'."
                )

    def plot_3d_plotly(self, trajectory_coords=None, trajectory_losses=None, 
                       title="Loss Landscape 3D", theme='dark', save_path=None,
                       color_by='loss', log_scale=False, show_floor_contours=True):
        """
        Generates an interactive 3D surface plot using Plotly, with an optional
        optimization path overlay.
        
        Args:
            trajectory_coords (list of tuple, optional): Coordinates (x, y) of the training trajectory.
            trajectory_losses (list of float, optional): Loss values corresponding to the trajectory.
            title (str): Title of the plot.
            theme (str): Plotly template, 'dark' (default) or 'light'.
            save_path (str, optional): Path to save the interactive HTML file.
            color_by (str): Coloring criteria for the surface: 'loss' (default) or 'gradient' (steepness).
            log_scale (bool): If True, applies log10 scaling to the loss values.
            show_floor_contours (bool): If True, projects a 2D contour map on the bottom floor.
            
        Returns:
            plotly.graph_objects.Figure: The generated Plotly figure object.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("Plotly is required for interactive 3D plots. Install it via 'pip install plotly'.")
            
        # Transpose the loss_grid to match Plotly's (y, x) row-col indexing convention
        z_surface = self.loss_grid.T
        
        # Determine minimum shift for log scale
        z_min = np.min(z_surface)
        shift = 0.0
        if log_scale:
            if z_min <= 0:
                shift = -z_min + 1.0
                z_surface = np.log10(z_surface + shift)
            else:
                z_surface = np.log10(z_surface)
        
        # Color mapping configuration
        if color_by == 'gradient':
            # Compute numerical gradient norm
            grad_y, grad_x = np.gradient(self.loss_grid)
            grad_norm = np.sqrt(grad_x**2 + grad_y**2)
            surfacecolor = grad_norm.T
            colorbar_title = "Gradient Norm (Steepness)"
            colorscale = 'Jet' if theme == 'dark' else 'Turbo'
        else:
            surfacecolor = z_surface
            colorbar_title = "Log10(Loss)" if log_scale else "Loss Value"
            colorscale = 'Viridis' if theme == 'dark' else 'Plasma'
            
        # Floor contours projection
        contours_config = {}
        if show_floor_contours:
            contours_config = dict(
                z=dict(
                    show=True,
                    usecolormap=True,
                    highlightcolor="white",
                    project=dict(z=True)
                )
            )
            
        # Specular reflection for metallic 3D feel
        lighting_config = dict(
            ambient=0.65,
            diffuse=0.85,
            roughness=0.4,
            specular=1.2,
            fresnel=0.4
        )
        
        fig = go.Figure()
        
        # Add surface trace
        fig.add_trace(go.Surface(
            x=self.x_coords,
            y=self.y_coords,
            z=z_surface,
            surfacecolor=surfacecolor,
            colorscale=colorscale,
            name='Loss Surface',
            contours=contours_config,
            lighting=lighting_config,
            colorbar=dict(title=colorbar_title, thickness=15),
            hovertemplate='X: %{x:.4f}<br>Y: %{y:.4f}<br>Z Value: %{z:.4f}<extra></extra>'
        ))
        
        # Add trajectory if provided
        if trajectory_coords is not None:
            traj_x = [pt[0] for pt in trajectory_coords]
            traj_y = [pt[1] for pt in trajectory_coords]
            
            # Use provided losses or default to average height
            if trajectory_losses is not None:
                traj_z = np.array(trajectory_losses)
                if log_scale:
                    traj_z = np.log10(traj_z + shift) if shift > 0.0 else np.log10(traj_z)
                traj_z = list(traj_z)
            else:
                traj_z = [float(np.mean(z_surface))] * len(trajectory_coords)
                
            steps = list(range(len(trajectory_coords)))
            
            # Path trace with timeline gradient
            fig.add_trace(go.Scatter3d(
                x=traj_x,
                y=traj_y,
                z=traj_z,
                mode='lines+markers',
                line=dict(
                    color='rgba(255, 255, 255, 0.4)' if theme == 'dark' else 'rgba(0, 0, 0, 0.3)',
                    width=4
                ),
                marker=dict(
                    size=5.5,
                    color=steps,
                    colorscale='YlOrRd', # Sequential progression coloring (initial steps are yellow, final minimum is red)
                    showscale=True,
                    colorbar=dict(
                        title="Optimizer Step",
                        thickness=10,
                        len=0.4,
                        x=1.15,
                        y=0.3
                    ),
                    line=dict(color='white', width=1)
                ),
                name='Optimizer Path',
                hovertemplate='Step: %{hovertext}<br>X: %{x:.4f}<br>Y: %{y:.4f}<br>Z Value: %{z:.4f}<extra></extra>',
                hovertext=[str(i) for i in range(len(trajectory_coords))]
            ))
            
            # Mark start and end points
            fig.add_trace(go.Scatter3d(
                x=[traj_x[0]], y=[traj_y[0]], z=[traj_z[0]],
                mode='markers',
                marker=dict(size=9, color='#FFCC00', symbol='diamond', line=dict(color='white', width=1.5)),
                name='Start (Init)',
                hovertemplate='Initialization Point<extra></extra>'
            ))
            
            fig.add_trace(go.Scatter3d(
                x=[traj_x[-1]], y=[traj_y[-1]], z=[traj_z[-1]],
                mode='markers',
                marker=dict(size=9, color='#00FFCC', symbol='square', line=dict(color='white', width=1.5)),
                name='End (Min)',
                hovertemplate='Final Point (Minimum)<extra></extra>'
            ))
            
        # Style the layout
        template = 'plotly_dark' if theme == 'dark' else 'plotly_white'
        font_color = 'white' if theme == 'dark' else 'black'
        grid_color = 'rgba(128,128,128,0.2)'
        z_title = 'Loss (Log10)' if log_scale else 'Loss'
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                y=0.95,
                font=dict(size=22, color=font_color)
            ),
            template=template,
            scene=dict(
                xaxis=dict(
                    title='Direction X',
                    gridcolor=grid_color,
                    zerolinecolor=grid_color,
                    backgroundcolor='rgba(0,0,0,0)' if theme == 'dark' else 'rgba(255,255,255,0)'
                ),
                yaxis=dict(
                    title='Direction Y',
                    gridcolor=grid_color,
                    zerolinecolor=grid_color,
                    backgroundcolor='rgba(0,0,0,0)' if theme == 'dark' else 'rgba(255,255,255,0)'
                ),
                zaxis=dict(
                    title=z_title,
                    gridcolor=grid_color,
                    zerolinecolor=grid_color,
                    backgroundcolor='rgba(0,0,0,0)' if theme == 'dark' else 'rgba(255,255,255,0)'
                ),
                aspectratio=dict(x=1, y=1, z=0.7),
                camera=dict(
                    eye=dict(x=1.4, y=1.4, z=1.0)
                )
            ),
            margin=dict(l=10, r=10, b=10, t=60),
            legend=dict(
                yanchor="top",
                y=0.95,
                xanchor="left",
                x=0.05
            )
        )
        
        if save_path is not None:
            # Ensure output dir exists
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.write_html(save_path)
            
        return fig
        
    def plot_3d_matplotlib(self, trajectory_coords=None, trajectory_losses=None,
                           title="Loss Landscape 3D", theme='light', save_path=None):
        """
        Generates a static 3D surface plot using Matplotlib.
        
        Args:
            trajectory_coords (list of tuple, optional): Coordinates (x, y) of the training trajectory.
            trajectory_losses (list of float, optional): Loss values corresponding to the trajectory.
            title (str): Title of the plot.
            theme (str): Visual style, 'dark' or 'light' (default).
            save_path (str, optional): Path to save the PNG file.
            
        Returns:
            matplotlib.figure.Figure: The generated Matplotlib figure object.
        """
        plt = self._import_matplotlib()
            
        # Set theme styles
        if theme == 'dark':
            plt.style.use('dark_background')
            text_color = 'white'
        else:
            plt.style.use('default')
            text_color = 'black'
            
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Grid for surface
        # Matplotlib expects X, Y to match Z in row/col.
        # Since self.X and self.Y are shape (len(y), len(x)) from meshgrid(x, y),
        # they match loss_grid.T which is (len(y), len(x))
        surf = ax.plot_surface(
            self.X, self.Y, self.loss_grid.T,
            cmap='viridis',
            edgecolor='none',
            alpha=0.8,
            antialiased=True
        )
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Loss Value')
        
        if trajectory_coords is not None and trajectory_losses is not None:
            traj_x = [pt[0] for pt in trajectory_coords]
            traj_y = [pt[1] for pt in trajectory_coords]
            traj_z = list(trajectory_losses)
            
            # Plot trajectory line and markers
            ax.plot(traj_x, traj_y, traj_z, color='red', marker='o', linewidth=3, markersize=5, label='Optimizer Path', zorder=10)
            ax.scatter([traj_x[0]], [traj_y[0]], [traj_z[0]], color='orange', marker='D', s=80, label='Start', zorder=11)
            ax.scatter([traj_x[-1]], [traj_y[-1]], [traj_z[-1]], color='cyan', marker='s', s=80, label='End (Min)', zorder=11)
            ax.legend()
            
        ax.set_title(title, fontsize=16, color=text_color)
        ax.set_xlabel('Direction X', color=text_color)
        ax.set_ylabel('Direction Y', color=text_color)
        ax.set_zlabel('Loss', color=text_color)
        
        # Customizing axes panes
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        plt.tight_layout()
        
        if save_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig

    def plot_contour_matplotlib(self, trajectory_coords=None, levels=25,
                                 title="Loss Landscape Contours", theme='light', save_path=None):
        """
        Generates a 2D contour plot using Matplotlib, useful for inspecting paths without 3D occlusion.
        
        Args:
            trajectory_coords (list of tuple, optional): Coordinates (x, y) of the training trajectory.
            levels (int): Number of contour levels to plot.
            title (str): Title of the plot.
            theme (str): Visual style, 'dark' or 'light' (default).
            save_path (str, optional): Path to save the PNG file.
            
        Returns:
            matplotlib.figure.Figure: The generated Matplotlib figure object.
        """
        plt = self._import_matplotlib()
            
        if theme == 'dark':
            plt.style.use('dark_background')
            text_color = 'white'
        else:
            plt.style.use('default')
            text_color = 'black'
            
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # Filled contour lines
        contour = ax.contourf(
            self.X, self.Y, self.loss_grid.T,
            levels=levels,
            cmap='viridis'
        )
        
        # Detailed line contours on top
        lines = ax.contour(
            self.X, self.Y, self.loss_grid.T,
            levels=levels,
            colors='black',
            linewidths=0.5,
            alpha=0.5
        )
        
        fig.colorbar(contour, ax=ax, label='Loss Value')
        
        if trajectory_coords is not None:
            traj_x = [pt[0] for pt in trajectory_coords]
            traj_y = [pt[1] for pt in trajectory_coords]
            
            # Plot 2D path
            ax.plot(traj_x, traj_y, color='red', marker='o', linewidth=2.5, markersize=4, label='Optimizer Path')
            ax.scatter([traj_x[0]], [traj_y[0]], color='orange', marker='D', s=70, label='Start', zorder=5)
            ax.scatter([traj_x[-1]], [traj_y[-1]], color='cyan', marker='s', s=70, label='End (Min)', zorder=5)
            ax.legend()
            
        ax.set_title(title, fontsize=15, color=text_color)
        ax.set_xlabel('Direction X', color=text_color)
        ax.set_ylabel('Direction Y', color=text_color)
        
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        if save_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig

    def plot_1d_matplotlib(self, alphas, values, value_name="Loss", title="1D Path Interpolation", theme='light', save_path=None):
        """
        Plots a 1D loss/metric interpolation path curve using Matplotlib.
        
        Args:
            alphas (array-like): 1D array of interpolation coordinates.
            values (array-like): 1D array of loss or metric values.
            value_name (str): Y-axis label name (e.g. "Loss" or "Accuracy").
            title (str): Title of the plot.
            theme (str): Visual style, 'dark' or 'light' (default).
            save_path (str, optional): Path to save the PNG file.
            
        Returns:
            matplotlib.figure.Figure: The generated Matplotlib figure object.
        """
        import os
        plt = self._import_matplotlib()
        
        if theme == 'dark':
            plt.style.use('dark_background')
            text_color = 'white'
            line_color = '#00FFCC'
            marker_color = '#FFCC00'
        else:
            plt.style.use('default')
            text_color = 'black'
            line_color = '#1f77b4'
            marker_color = '#ff7f0e'
            
        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.plot(alphas, values, color=line_color, marker='o', linestyle='-', linewidth=2, markersize=5, label=value_name)
        
        ax.set_title(title, fontsize=15, color=text_color)
        ax.set_xlabel('Interpolation Parameter (Alpha)', color=text_color)
        ax.set_ylabel(value_name, color=text_color)
        
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
