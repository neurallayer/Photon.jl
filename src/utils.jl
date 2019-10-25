function getEntry(d, val)
    for (k,v) in d
        if v == val return k end
    end
    return ""
end

function weights(layer, d=Dict(), root="", mode=0)
    c = typeof(layer)
    for name in fieldnames(c)
        if ! isdefined(layer, name)
            println("undefined found: ", name)
            continue
        end
        field = getfield(layer,name)
        if field isa Param
            fqn = "$root:$c:$name:$(size(field))"
            if mode == 0
                old_key = getEntry(d,field)
                if old_key != ""
                    d[fqn] = old_key
                else
                    d[fqn] = field
                end
            else
                val = d[fqn]
                if val isa String
                    val = d[val]
                end
                print("set value for $fqn")
                    # Set the value
            end
            println("$root:$c:$name:$(size(field))")
        elseif (field isa Array) || (field isa Tuple)
            for (idx, elem) in enumerate(field)
                weights(elem, d, "$root:$c:$name:$idx", mode)
            end
        else
            weights(field, d, "$root:$c:$name", mode)
        end
    end
	d
end



"""
Beginning of allowing for a single model instance to run on multiple devices
(expiremental)
"""
function move2gpu(X)
	isa(X, Array) ? KnetArray(X) : X
end

function move2cpu(X)
	isa(X, KnetArray) ? Array(X) : X
end
