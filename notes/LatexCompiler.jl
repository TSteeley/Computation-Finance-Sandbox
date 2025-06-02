using HTTP, Sockets, JLD2

# Latex compilers code
# bibtex   => b
# xelatex  => x
# pdflatex => p
# lualatex => l

# =========================
# =====   Compilers   =====
# =========================

functions = 

compilers(settings) = Dict(
    'b' => `bibtex $(settings["PROJECT_FOLDER"])/$(settings["AUX_FOLDER"])/$(settings["MAIN_FILE"])`,
    'x' => `xelatex --aux-directory=$(settings["PROJECT_FOLDER"])/$(settings["AUX_FOLDER"]) --output-directory=$(settings["PROJECT_FOLDER"])/$(settings["AUX_FOLDER"]) $(settings["MAIN_FILE"]).tex`,
    'p' => `pdflatex --aux-directory=$(settings["PROJECT_FOLDER"])/$(settings["AUX_FOLDER"]) --output-directory=$(settings["PROJECT_FOLDER"])/$(settings["AUX_FOLDER"]) $(settings["MAIN_FILE"]).tex`,
    'l' => `lualatex --aux-directory=$(settings["PROJECT_FOLDER"])/$(settings["AUX_FOLDER"]) --output-directory=$(settings["PROJECT_FOLDER"])/$(settings["AUX_FOLDER"]) $(settings["MAIN_FILE"]).tex`,
)

function handler(req::HTTP.Request)
    # Read the file as bytes
    global settings 
    file_data = read("$(settings["MAIN_FILE"]).pdf")
    
    # Respond with the file and set the appropriate content type
    return HTTP.Response(200, ["Content-Type" => "application/pdf"], file_data)
end

# Start the HTTP server
function start_server()
    try 
        cp("./$(settings["PROJECT_FOLDER"])/$(settings["AUX_FOLDER"])/$(settings["MAIN_FILE"]).pdf", "./$(settings["MAIN_FILE"]).pdf",force=true)
        println("Serving file at http://localhost:8080")
        @async HTTP.serve(handler, "0.0.0.0", 8080)
    catch
        println("No document to display")
    end
end

function PDF_Compile(settings)
    try
        for c in settings["FULL_RECIPE"]
            run(compilers(settings)[c])
        end
    catch
        println("Compilation failed")
    end
end

function quick_Compile(settings)
    try
        for c in settings["FAST_RECIPE"]
            run(compilers(settings)[c])
        end
    catch
        println("Compilation failed")
    end
end

function main()
    settings = 
    success=false # for-if-end at home
    for (root, dirs, files) in walkdir(".")
        if any("settings.jld2" .== files)
            # println("Found in $(root)")
            @load "$root/settings.jld2" settings
            global settings
            success = true
        end
    end
    if .! success
        settings = Dict{String, String}()
        settings["FULL_RECIPE"] = "xbxx"
        settings["FAST_RECIPE"] = "x"
        settings["PROJECT_FOLDER"] = "Project"
        settings["AUX_FOLDER"] = "auxFiles"
        settings["MAIN_FILE"] = "main"
        global settings

        mkdir("./$(settings["PROJECT_FOLDER"])")
        mkdir("./$(settings["PROJECT_FOLDER"])/$(settings["AUX_FOLDER"])")

        @save "$(settings["PROJECT_FOLDER"])/settings.jld2" settings
    end

    while true 
        print("PDF Compiler> ")
        inp = readline()

        if inp == "h"
            println("Options:")
            println("    \"c\" => Compile main.tex")
            println("    \"f\" => Fast compile main.tex")

        elseif inp == "c"
            global settings
            PDF_Compile(settings)
            start_server()
        elseif inp == "f"
            global settings
            quick_Compile(settings)
            start_server()
        end
    end
end

main()