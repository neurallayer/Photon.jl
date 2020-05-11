
export plotmetrics


function getMetricsAtEpoch(m,ep)
      s = intersect(Set(ep[1]), Set(m[1]))
      a = [v for (k,v) in zip(ep...) if k in s]
      b = [v for (k,v) in zip(m...) if k in s]
      return (a,b)
end

"""
Plot the metrics after some training. This function will plot all the metrics
in a single graph.

In order to avoid Photon being dependend on Plots, the calling code will have
to provide that module as the first parameter.

# Usage

```julia
train!(workout, mydata, epochs=10)

import Plots
plotmetrics(Plots, workout)
```
"""
function plotmetrics(Plots::Module, workout::Workout, metrics=[:loss, :val_loss]; epoch_only=false)
      p = nothing
      if epoch_only
            ep = history(workout, :epoch)
      end

      for (idx, metric) in enumerate(metrics)
            h = history(workout, metric)

            if epoch_only
                  h = getMetricsAtEpoch(h, ep)
            end

            if idx == 1
                  xlabel = epoch_only ? "epochs" : "steps"
                  p = Plots.plot(h..., xlabel = xlabel, ylabel="values", label=metric)
            else
                  Plots.plot!(h..., label=metric)
            end
      end
      return p # otherwise no plotting in Juno
end

"""
Move the batch from the first to the last dimension
"""
function batchlast(a::AbstractArray)
      p = collect(1:length(size(a)))
      push!(p, popfirst!(p))
      permutedims(a,p)
end


"""
Move the batch from the last to the first dimension
"""
function batchfirst(a::AbstractArray)
      p = collect(1:length(size(a)))
      pushfirst!(p, pop!(p))
      permutedims(a,p)
end
