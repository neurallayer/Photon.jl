
abstract type Callback end

"""
A callback should implement the following function signature.
"""
function (::Callback)(workout::Workout, phase::Symbol) end


"""
Save the workout at the end of every epoch. Optionally provide a filename.

# Usage
```julia
train!(workout, data, cb=EpochSave())
```
"""
struct EpochSave <: Callback
    filename::Union{String, Nothing}
    EpochSave(filename=nothing) = new(filename)
end

function (c::EpochSave)(workout::Workout, phase::Symbol)
    if phase == :valid
        if c.filename !== nothing
            saveworkout(workout, c.filename)
        else
            saveworkout(workout)
        end
    end
end


"""
Save the Workout if a certain metric has improved since the last epoch

# Usage
```julia

# save as long as the validation loss is declining
train!(workout, data, cb=AutoSave(:val_loss))
```

"""
mutable struct AutoSave <: Callback
    value::Float64
    metric::Symbol
    filename::Union{String, Nothing}
    mode::Symbol
    function AutoSave(metric::Symbol, filename=nothing, mode=:min)
        init_value = mode == :min ? Inf : -Inf
        new(init_value, metric, filename, mode)
    end
end

function (c::AutoSave)(workout::Workout, phase::Symbol)
    if phase == :valid
        getmetricvalue(workout, c.metric) do x
            condition = c.mode == :min ? x < c.value : x > c.value
            if condition
                if c.filename !== nothing
                    saveworkout(workout, c.filename)
                else
                    saveworkout(workout)
                end
                c.value = x
            end
        end
    end
end


"""
Stop the training if a certain metric didn't improve
"""
mutable struct EarlyStop <: Callback
    value::Float64
    metric::Symbol
    mode::Symbol
    function EarlyStop(metric::Symbol, mode=:min)
        init_value = mode == :min ? Inf : -Inf
        new(init_value, metric, mode)
    end
end

function (c::EarlyStop)(workout::Workout, phase::Symbol)
    if phase == :valid
        getmetricvalue(workout, c.metric) do x
            condition = c.mode == :min ? x < c.value : x > c.value
            if condition
                stop(workout,"$(c.metric) didn't improve anymore")
            else
                c.value = x
            end
        end
    end
end




"""
A meter is responsible for presenting metric values. It can be used just as
a regular callback argument to the *train!* function.
A meter is not limited to printing results to the console output, it can also be
showing it on a TensorBoard or storing results in a database for example.
"""
abstract type Meter <: Callback end



"""
Use a SilentMeter in case no output is required. The default of
*train!* function is to use a ConsoleMeter and using the SilentMeter this
behavior can be overriden.

# Usage

```julia
train!(workout, data, val_data; epochs=5, cb=SilentMeter())
```
"""
struct SilentMeter <: Meter
end

(meter::SilentMeter)(workout::Workout, phase::Symbol) = ()



"""
Logs metrics to the console output. By default it will only log at the end of an epoch
and log the epoch, step, loss and validation loss. This is also the default configuration
when you run train! without specifying oter callbacks.

# Usage

```julia
meter = ConsoleMeter([:loss, :val_accuracy, :accuracy]; epochOnly=false)
train!(workout, data, val_data; epochs=5, cb=meter)
```
"""
mutable struct ConsoleMeter <: Meter
    throttle::Float64
    next::Float64
    metricnames::Vector{Symbol}
    epochOnly

    function ConsoleMeter(metrics=[:loss, :val_loss]; throttle=1.0, epochOnly=true)
        new(throttle, 0.0, metrics, epochOnly)
    end
end


function (meter::ConsoleMeter)(workout::Workout, phase::Symbol)
    meter.epochOnly && phase == :train && return
    now = time()
    if now > meter.next || phase == :valid
        result = ""
        for metricname in meter.metricnames
            getmetricvalue(workout, metricname) do value
                    result *= @sprintf  " - %s: %1.4f" metricname value
            end
        end
        if result != ""
            s = @sprintf "[%4d:%7d]" workout.epochs workout.steps
            print("\r", s, result)
            meter.next = now + meter.throttle
        end
        phase == :valid && println()
    end
end


"""
Logs metrics to a TensorBoard file so it can be viewed with TensorBoard. By default
it will log the metrics loss and val_loss at the end of each training step
and end of the validation phase.

This meter depends on the TensorBoardLogger to be installed. So if you didn't do so
already, please run:

    import Pkg; Pkg.add("TensorBoardLogger")
"""
mutable struct TensorBoardMeter <: Meter
    logger
    path::String
    metrics
    last_processed::IdDict{Symbol,Int}

    function TensorBoardMeter(path="./tensorboard_logs/runs", metrics=[:loss, :val_loss])
        try
            @eval import TensorBoardLogger
        catch
            @warn "Package TensorBoardLogger not installed"
        end
        new(nothing, path, metrics, IdDict())
    end
end

function (meter::TensorBoardMeter)(workout::Workout, phase::Symbol)
    if meter.logger === nothing
        meter.logger = TensorBoardLogger.TBLogger(meter.path)
    end

    for metric in meter.metrics
        getmetricvalue(workout, metric) do value
            last = get(meter.last_processed, metric, -1)
            workout.steps > last && TensorBoardLogger.log_value(meter.logger,
                string(metric), value, step=workout.steps)
            meter.last_processed[metric] = workout.steps
        end
    end
end


"""
Logs metrics to a delimeted text file. By default it will log the metrics loss and
val_loss at the end of each training step and end of the validation phase.

"""
mutable struct FileMeter <: Meter
    fileio
    filename::String
    metrics
    last_processed::IdDict{Symbol,Int}
    epochOnly::Bool

    function FileMeter(filename="traininglog.txt", metrics=[:loss, :val_loss], epochOnly=true)
        try
            @eval import DelimitedFiles
        catch
            @warn "Package DelimitedFiles not installed"
        end
        io = open(filename, "w")
        new(io, filename, metrics, IdDict(), epochOnly)
    end
end

function (meter::FileMeter)(workout::Workout, phase::Symbol)
    meter.epochOnly && phase == :train && return
    for metric in meter.metrics
        getmetricvalue(workout, metric) do value
            last = get(meter.last_processed, metric, -1)
            workout.steps > last && DelimitedFiles.writedlm(meter.fileio,
                [(metric, value, workout.steps)])
            meter.last_processed[metric] = workout.steps
        end
    end
    flush(meter.fileio)
end


"""
Plot metrics using the Plots module. At the end of each epoch the plot will
be updated with the values of the metrics. This works especially nice if you
are prototyping some model in an IDE like Juno.
"""
struct PlotMeter <: Meter
    plt
    metrics::Vector{Symbol}
    Plots::Module

    function PlotMeter(Plots::Module, metrics=[:loss, :val_loss])
        plt = Plots.plot(length(metrics),xlabel = "steps", ylabel="values", label=metrics)
        Plots.display(plt)
        new(plt, metrics, Plots)
    end
end


function (m::PlotMeter)(workout::Workout, phase::Symbol)
    phase == :train && return
    for (i, metric) in enumerate(m.metrics)
        if hasmetric(workout, metric)
            m.plt[i] = history(workout, metric)
        end
    end
    m.Plots.display(m.plt)
end
