
export plotmetrics


"""
Plot the metrics after some training. This function will plot all the metrics
in a single graph.

In order to avoid Photon being dependend on Plots, the calling code will have
to provide that module as the first parameter.

# Usage

```julia
fit!(workout, mydata, epochs=10)

import Plots
plotmetrics(Plots, workout)
```
"""
function plotmetrics(Plots::Module, workout::Workout, metrics=[:loss, :val_loss])
      p = nothing

      for (idx, metric) in enumerate(metrics)
            h = history(workout, metric)

            if idx == 1
                  p = Plots.plot(h..., xlabel = "steps", ylabel="values", label=metric)
            else
                  Plots.plot!(h..., label=metric)
            end
      end
      return p # otherwise no plotting in Juno
end
