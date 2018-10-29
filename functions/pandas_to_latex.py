import io

# =============================================================================
#  Convert pandas table to latex
# =============================================================================
def dataframe_to_latex(dataframe,
                       label = None,
                       caption = None,
                       italic = None):
    """This function calculates the stimulus current at the current source for
    a single monophasic pulse stimulus at each point of time
    
    Parameters
    ----------
    model : time
        Lenght of one time step.
    dt : string
        Describes, how the ANF is stimulated; either "internal" or "external" is possible
    phase_durations : string
        Describes, which pulses are used; either "mono" or "bi" is possible
    start_interval : time
        Time until (first) pulse starts.
    delta : time
        Time which is still simulated after the end of the last pulse
    stimulation_type : amp/[k_noise]
        Is multiplied with k_noise.
    pulse_form : amp/[k_noise]
        Is multiplied with k_noise.
    
    Returns
    -------
    current matrix
        Gives back a vector of currents for each timestep
    runtime
        Gives back the duration of the simulation
    """
    
    ##### get number of columns and rows
    nof_cols = dataframe.shape[1]
    nof_rows = dataframe.shape[0]
    
    ##### get range of none italic columns
    if italic is None:
        normal_range = range(nof_rows)
        italic_range = range(0)
    else:
        normal_range = range(0, italic[0])
        italic_range = italic
    
    ##### initialize output
    output = io.StringIO()
    
    ##### define column format/alignment
    colFormat = ("%s|%s" % ("l", "c" * nof_cols))
    
    ##### Write table header
    output.write("\\begin{table}[htb]\n")
    output.write("\\centering\n")
    output.write("\\begin{tabular}{%s}\n" % colFormat)
    columnLabels = ["%s" % label for label in dataframe.columns]
    output.write("& %s\\\\\\hline\n" % " & ".join(columnLabels))
    
    ##### Write data lines (no italic)
    for ii in normal_range:
        output.write("%s & %s\\\\\n"
                     % (dataframe.index[ii], " & ".join([str(val) for val in dataframe.iloc[ii]])))
    
    ##### Write data lines (italic)
    for ii in italic_range:
        output.write("\\textit{%s} & %s\\\\\n"
                     % (dataframe.index[ii], " & ".join(["\\textit{%s}" % str(val) for val in dataframe.iloc[ii]])))
    
    ##### Write footer
    output.write("\\end{tabular}\n")
    if caption is not None:
        output.write("\\caption{%s}\n" % caption)
    if label is not None:
        output.write("\\label{%s}\n" % label)
    output.write("\\end{table}")
    
    ##### save output in new variable
    table_string = output.getvalue()
    
    ##### replace % with /%
    table_string = table_string.replace('%','\%')
        
    ##### replace u a mikro sign
    table_string = table_string.replace('(u','($\mu$')
     
    return table_string

