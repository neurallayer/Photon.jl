
export plotmetrics


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
