import io

# =============================================================================
#  Convert pandas table to latex
# =============================================================================
def dataframe_to_latex(dataframe,
                       label = None,
                       caption_top = None,
                       caption_bottom = None,
                       italic = None,
                       vert_line = None,
                       upper_col_names = None):
    """This function converts a pandas dataframe in latex table code.
    
    Example for using the function:
    
        with open("path/my_table.tex", "w") as tf:
            tf.write(dataframe_to_latex(my_table, label = "tbl:table_caption"))
    
    The table can then be included to a latex script with:
    
        \input{path/my_table}
    
    Parameters
    ----------
    dataframe : pandas dataframe
        Dataframe to be converted.
    label : string
        Specifies label of the latex table
    caption_top : string
        Specifies caption above the latex table
    caption_bottom : string
        Specifies caption below the latex table
    italic : list of integers
        The list specifies the numbers of the rows, which should be written
        in italics..
    vert_line : list of integers
        The list specifies the column numbers where a vertical line is put
        afterwards. Counting starts from zero.
    upper_col_names : list of strings
        The list specifies the names of a second column name row above the usual
        column name row.
    
    Returns
    -------
    string
        Gives back a string with latex code.
    """
    
    ##### get number of columns and rows
    nof_rows, nof_cols = dataframe.shape
    
    ##### get range of none italic columns
    if italic is None:
        italic_rows = []
    else:
        italic_rows = italic
    
    ##### initialize output
    output = io.StringIO()
    
    ##### define column format/alignment
    if vert_line == None:
        colFormat = ("%s|%s" % ("l", "c" * nof_cols))
    else:
        colFormat = "l|"
        for ii in range(0,nof_cols):
            colFormat = colFormat + "c"
            if ii in vert_line: colFormat = colFormat + "|"
    
    ##### Write table header
    output.write("\\begin{table}[htb]\n")
    output.write("\\centering\n")
    if caption_top is not None: output.write("\\caption{%s}\n" % caption_top)
    output.write("\\begin{tabular}{%s}\n" % colFormat)
    if upper_col_names is not None: output.write("& \\multicolumn{3}{c|}{%s}\\\\\n" % "} & \\multicolumn{3}{c}{".join(upper_col_names))
    columnLabels = ["%s" % label for label in dataframe.columns]
    output.write("& %s\\\\\\hline\n" % " & ".join(columnLabels))
    
    ##### Write data rows
    for ii in range(nof_rows):
        if ii in italic_rows:
            ##### italic row
            output.write("\\textit{%s} & %s\\\\\n"
                     % (dataframe.index[ii], " & ".join(["\\textit{%s}" % str(val) for val in dataframe.iloc[ii]])))
        else:
            ##### normal row
            output.write("%s & %s\\\\\n"
                     % (dataframe.index[ii], " & ".join([str(val) for val in dataframe.iloc[ii]])))
    
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
        
    ##### replace u with mikro sign
    table_string = table_string.replace('(u','($\mu$')
     
    return table_string

