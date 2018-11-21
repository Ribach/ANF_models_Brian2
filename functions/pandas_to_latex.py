import io

# =============================================================================
#  Convert pandas table to latex
# =============================================================================
def dataframe_to_latex(dataframe,
                       label = None,
                       caption_top = None,
                       caption_bottom = None,
                       italic = None):
    """This function converts a pandas dataframe in latex table code.
    
    Parameters
    ----------
    dataframe : pandas dataframe
        Dataframe to be converted.
    label : string
        If specified, it defines the label of the latex tabe
    caption : string
        If specified, it defines the caption of the latex tabe
    italic : list of integers
        The list specifies the row numbers of the rows, which should be written
        in italics..
    
    Returns
    -------
    string
        Gives back a string with latex code.
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
    if caption_top is not None: output.write("\\caption{%s}\n" % caption_top)
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
    if caption_bottom is None:
        output.write("\\end{tabular}\n")
        
    if caption_top is None and caption_bottom is not None:
        output.write("\\end{tabular}\n")
        output.write("\\caption{%s}\n" % caption_bottom)
        
    if caption_top is not None and caption_bottom is  not None:
        output.write("\\hline\n")
        output.write("\\end{tabular}\n")
        output.write("\\caption*{%s}\n" % caption_bottom)

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

