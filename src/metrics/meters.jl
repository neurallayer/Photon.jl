
export TensorBoardMeter, ConsoleMeter, FileMeter, PlotMeter


"""
Logs metrics to the console output. By default it will only log at the end of an epoch
and log the epoch, step, loss and validation loss. This is also the default configuration
when you run fit! without specifying your own meters.

Example:

    meter = ConsoleMeter([:loss, :val_accuracy, :accuracy]; epochOnly=false)
    fit!(workout, data, val_data; epochs=5, meters=[meter])

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


function display(meter::ConsoleMeter, workout::Workout, phase::Symbol)
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

function display(meter::TensorBoardMeter, workout::Workout, phase::Symbol)
    if meter.logger == nothing
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

function display(meter::FileMeter, workout::Workout, phase::Symbol)
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
Plot metrics using Plots module
"""
struct PlotMeter
    plt
    metrics::Vector{Symbol}
    Plots::Module

    function PlotMeter(Plots::Module, metrics=[:loss, :val_loss])
        plt = Plots.plot(length(metrics),xlabel = "steps", ylabel="values", label=metrics)
        Plots.display(plt)
        new(plt, metrics, Plots)
    end
end


function display(m::PlotMeter, workout::Workout, phase::Symbol)
    phase == :train && return
    for (i, metric) in enumerate(m.metrics)
        if hasmetric(workout, metric)
            m.plt[i] = history(workout, metric)
        end
    end
    m.Plots.display(m.plt)
end
