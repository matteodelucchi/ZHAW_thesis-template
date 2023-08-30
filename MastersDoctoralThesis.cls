%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Masters/Doctoral Thesis ZHAW
% Class File
% Version 1.7 (24/8/20)
% 
% ZHAW class version downloaded from:
% https://github.com/matteodelucchi/ZHAW_thesis-template
% 
% University specific changes released in Version 1.7 were made by:
% Matteo Delucchi
%
% Original class was downloaded from:
% http://www.LaTeXTemplates.com
% 
% Authors:
% Vel (vel@latextemplates.com)
% Johannes Böttcher
%
% Notes:
% 1) This class file defines the structure and layout of the template file (main.tex).
% 2) It has been written in such a way that under most circumstances you should not need
% to edit it; updating it to a newer version will be harder. If you do make changes, please change the name of
% the file and add comments to make your changes more visible.
%
% Class license:
% LPPL v1.3c (http://www.latex-project.org/lppl)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%----------------------------------------------------------------------------------------
%   CLASS DEFINITION AND PARAMETERS
%----------------------------------------------------------------------------------------

\NeedsTeXFormat{LaTeX2e}[1996/12/01]
\newcommand{\classname}{MastersDoctoralThesis}
\ProvidesClass{\classname}[2020/08/24 v1.7 LaTeXTemplates.com]
\providecommand{\baseclass}{book}
\RequirePackage{etoolbox}
\RequirePackage{xparse}
\newbool{nolistspace}
\newbool{chapteroneline}
\newbool{listtoc}
\newbool{toctoc}
\newbool{parskip}
\newbool{hyperrefsupport}
\booltrue{hyperrefsupport}
\newbool{headsepline}
\newbool{consistentlayout}
\newbool{helveticafont}

\DeclareOption{nohyperref}{\boolfalse{hyperrefsupport}}
\DeclareOption{nolistspacing}{\booltrue{nolistspace}}
\DeclareOption{liststotoc}{\booltrue{listtoc}}
\DeclareOption{chapterinoneline}{\booltrue{chapteroneline}}
\DeclareOption{toctotoc}{\booltrue{toctoc}}
\DeclareOption{parskip}{\booltrue{parskip}}
\DeclareOption{headsepline}{\booltrue{headsepline}}
\DeclareOption{consistentlayout}{\booltrue{consistentlayout}}
\DeclareOption{helveticafont}{\booltrue{helveticafont}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{\baseclass}}

\ProcessOptions\relax

\LoadClass{\baseclass}

%  Simple interface for the user  to customize the chapter titles 
\ProvideDocumentCommand{\abovechapterskip}{}{\vspace*{20pt}}
\ProvideDocumentCommand{\chapterbelowskip}{}{\vspace*{40pt}}
\ProvideDocumentCommand{\chapterinbetweenskip}{}{\vspace*{20pt}}
\ProvideDocumentCommand{\autodot}{}{}
\ProvideDocumentCommand{\mdtChapapp}{}{}
\ProvideDocumentCommand{\chapteralign}{}{\raggedright}
\ProvideDocumentCommand{\chapterfont}{}{\Huge\bfseries}
\ProvideDocumentCommand{\chapterprefixfont}{}{\LARGE\bfseries}
\DeclareDocumentCommand{\@makechapterhead}{ m }{%
    \abovechapterskip
    {\parindent \z@ \chapteralign \normalfont
        \ifnum \c@secnumdepth >\m@ne
        \if@mainmatter
        \ifbool{chapteroneline}{%
            \chapterfont \mdtChapapp\thechapter\autodot\enspace
        }{%
            \chapterprefixfont \@chapapp\space \thechapter
            \par\nobreak
            \chapterinbetweenskip
        }%
        \fi
        \fi
        \interlinepenalty\@M%
        \chapterfont #1\par\nobreak
        \chapterbelowskip
    }
    \thispagestyle{\chapter@p@gestyle}
}
\def\@makeschapterhead#1{%
    \abovechapterskip
    {\parindent \z@ \chapteralign
        \normalfont
        \interlinepenalty\@M
        \chapterfont  #1\par\nobreak
        \chapterbelowskip
    }
    \thispagestyle{\chapter@p@gestyle}
}


% Addchap provides unnumbered chapters with an entry in the table of contents as well as an updated header
\ProvideDocumentCommand{\addchap}{ s o m }{%
    \chapter*{#3}%
    \markboth{}{}%
    \IfBooleanTF{#1}{%
    }{%
        \IfNoValueTF{#2}{%
            \addchaptertocentry{#3}%
            \markboth{\MakeMarkcase{#3}}{\MakeMarkcase{#3}}%
        }{%
            \addchaptertocentry{#2}%
            \markboth{\MakeMarkcase{#2}}{\MakeMarkcase{#2}}%
        }%
    }%
}%

\ProvideDocumentCommand{\addsec}{ s o m }{%
    \section*{#3}%
    \markright{}%
    \IfBooleanTF{#1}{%
    }{%
        \IfNoValueTF{#2}{%
            \addcontentsline{toc}{section}{#3}%
            \markright{\MakeMarkcase{#3}}%%
        }{%
            \addcontentsline{toc}{section}{#2}%
            \markright{\MakeMarkcase{#2}}%
        }%
    }%
}%

%----------------------------------------------------------------------------------------
%   CLASS OPTIONS
%----------------------------------------------------------------------------------------

\ifbool{parskip}{\RequirePackage{parskip}} % If the parskip option is passed to the class, require the parskip package 


\ifbool{listtoc}{% If the liststotoc option has been passed to the class, add the lists to the table of contents
    \patchcmd{\listoftables}{\@starttoc{lot}}{%
        \addchaptertocentry{\listtablename}\@starttoc{lot}%
    }{}{}%
    \patchcmd{\listoffigures}{\@starttoc{lof}}{%
        \addchaptertocentry{\listfigurename}\@starttoc{lof}%
    }{}{}%
}

\ifbool{toctoc}{% If the toctotoc options has been passed to the class, add the table of contents to the table of contents
    \patchcmd{\tableofcontents}{\@starttoc{toc}%
}{%
    \addchaptertocentry{\contentsname}\@starttoc{toc}}{}{}%
}

\patchcmd{\tableofcontents}{\MakeUppercase}{\MakeMarkcase}{}{}
\patchcmd{\tableofcontents}{\MakeUppercase}{\MakeMarkcase}{}{}
\patchcmd{\listoffigures}{\MakeUppercase}{\MakeMarkcase}{}{}
\patchcmd{\listoffigures}{\MakeUppercase}{\MakeMarkcase}{}{}
\patchcmd{\listoftables}{\MakeUppercase}{\MakeMarkcase}{}{}
\patchcmd{\listoftables}{\MakeUppercase}{\MakeMarkcase}{}{}

% If the option `nolistspacing' is given, the spacing in the different lists is reduced to single spacing. This option is only useful, if the spacing of the document has been changed to onehalfspacing or doublespacing.
\ifbool{nolistspace}{
    \patchcmd{\listoffigures}{%
        \@starttoc{lof}
    }{%
        \begingroup%
        \singlespace\@starttoc{lof}\endgroup%
    }{}{}%
    \patchcmd{\listoftables}{%
        \@starttoc{lot}
    }{%
        \begingroup%
        \singlespace\@starttoc{lot}\endgroup%
    }{}{}%
    \patchcmd{\tableofcontents}{%
        \@starttoc{toc}
    }{%
        \begingroup%
        \singlespace\@starttoc{toc}\endgroup%
    }{}{}%
}{}


%----------------------------------------------------------------------------------------
%   REQUIRED PACKAGES
%----------------------------------------------------------------------------------------

\RequirePackage{babel} % Required for automatically changing names of document elements to languages besides english

\RequirePackage{scrbase} % Required for handling language-dependent names of sections/document elements

% Load package scrhack, which provides fixes for various packages.
% Defer the loading, because scrhack should be loaded after other packages.
% See here for the problem: https://tex.stackexchange.com/questions/659887/
\AddToHook{begindocument/before}{\usepackage{scrhack}}
%\RequirePackage{scrhack}

\RequirePackage{setspace} % Required for changing line spacing

\RequirePackage{longtable} % Required for tables that span multiple pages (used in the symbols, abbreviations and physical constants pages)

\RequirePackage{siunitx} % Required for \SI commands

\RequirePackage{graphicx} % Required to include images
\graphicspath{{Figures/}{./}} % Specifies where to look for included images

\RequirePackage{booktabs} % Required for better table rules

\RequirePackage{caption} % Required for customising the captions
\captionsetup{justification=centerlast,font=small,labelfont=sc,margin=50pt}



%----------------------------------------------------------------------------------------
%   DEFINE CUSTOM THESIS INFORMATION COMMANDS
%----------------------------------------------------------------------------------------

\NewDocumentCommand{\thesistitle} { o m }{%
 \IfValueTF{#1}{\def\shorttitle{#1}}{\def\shorttitle{#2}}%
 \def\@title{#2}%
 \def\ttitle{#2}%
}
\NewDocumentCommand{\thesistype}{m}{\newcommand{\ttype}{#1}}
\NewDocumentCommand{\thesisdate}{m}{\newcommand{\tdate}{#1}}
\DeclareDocumentCommand{\author}{m}{\newcommand{\authorname}{#1}\renewcommand{\@author}{#1}}
\NewDocumentCommand{\supervisorA}{m}{\newcommand{\supnameA}{#1}}
\NewDocumentCommand{\supervisorAmail}{m}{\newcommand{\supmailA}{#1}}
\NewDocumentCommand{\supervisorAweb}{m}{\newcommand{\supwebA}{#1}}
\NewDocumentCommand{\supervisorAinfo}{m}{\newcommand{\supinfoA}{#1}}
\NewDocumentCommand{\supervisorB}{m}{\newcommand{\supnameB}{#1}}
\NewDocumentCommand{\supervisorBmail}{m}{\newcommand{\supmailB}{#1}}
\NewDocumentCommand{\supervisorBweb}{m}{\newcommand{\supwebB}{#1}}
\NewDocumentCommand{\supervisorBinfo}{m}{\newcommand{\supinfoB}{#1}}
\NewDocumentCommand{\examiner}{m}{\newcommand{\examname}{#1}}
\NewDocumentCommand{\degree}{m}{\newcommand{\degreename}{#1}}
\NewDocumentCommand{\studyprogram}{m}{\newcommand{\studyprog}{#1}}
\NewDocumentCommand{\studyprogramlink}{m}{\newcommand{\studyproglink}{#1}}
\NewDocumentCommand{\university}{m}{\newcommand{\univname}{#1}}
\NewDocumentCommand{\universitygerman}{m}{\newcommand{\univnameger}{#1}}
\NewDocumentCommand{\department}{m}{\newcommand{\deptname}{#1}}
\NewDocumentCommand{\institute}{m}{\newcommand{\instname}{#1}}
\NewDocumentCommand{\group}{m}{\newcommand{\groupname}{#1}}
\NewDocumentCommand{\subject}{m}{\newcommand{\subjectname}{#1}}
\NewDocumentCommand{\keywords}{m}{\newcommand{\keywordnames}{#1}}

\NewDocumentCommand{\universitylink}{m}{\newcommand{\univlink}{#1}}
\NewDocumentCommand{\universitylinkgerman}{m}{\newcommand{\univlinkger}{#1}}
\NewDocumentCommand{\departmentlink}{m}{\newcommand{\deptlink}{#1}}
\NewDocumentCommand{\institutelink}{m}{\newcommand{\instlink}{#1}}
\NewDocumentCommand{\grouplink}{m}{\newcommand{\grplink}{#1}}



\newcommand{\checktoopen}{% New command to move content to the next page which prints to the next odd page if twosided mode is active  
    \if@openright\cleardoublepage\else\clearpage\fi
    \ifdef{\phantomsection}{\phantomsection}{}% The \phantomsection command is necessary for hyperref to jump to the correct page
}

\NewDocumentCommand{\bhrule}{}{\typeout{--------------------}}
\NewDocumentCommand{\tttypeout}{m}{\bhrule\typeout{\space #1}\bhrule}

\newcommand{\HRule}{\rule{.9\linewidth}{.6pt}} % New command to make the lines in the title page
\newcommand{\decoRule}{\rule{.8\textwidth}{.4pt}} % New command for a rule to be used under figures

\setcounter{tocdepth}{3} % The depth to which the document sections are printed to the table of contents
\ProvideDocumentCommand{\addchaptertocentry}{ m }{%
    \addcontentsline{toc}{chapter}{#1}%
}

%----------------------------------------------------------------------------------------
%   FONT
%----------------------------------------------------------------------------------------

\ifbool{helveticafont}{
    \RequirePackage{helvet}
    \renewcommand\familydefault{\sfdefault} 
    \usepackage[T1]{fontenc}
}{
    % nothing
}

%----------------------------------------------------------------------------------------
%   COLOURS
%----------------------------------------------------------------------------------------

\usepackage{xcolor} % Required for specifying custom colours

\colorlet{mdtRed}{red!50!black}

%----------------------------------------------------------------------------------------
%   MARGINS
%----------------------------------------------------------------------------------------

\RequirePackage{geometry}
\geometry{
    headheight=4ex,
    includehead,
    includefoot
}

\raggedbottom

%----------------------------------------------------------------------------------------
%   PENALTIES
%----------------------------------------------------------------------------------------

\doublehyphendemerits=10000 % No consecutive line hyphens
\brokenpenalty=10000 % No broken words across columns/pages
\widowpenalty=9999 % Almost no widows at bottom of page
\clubpenalty=9999 % Almost no orphans at top of page
\interfootnotelinepenalty=9999 % Almost never break footnotes

%----------------------------------------------------------------------------------------
%   HEADERS AND FOOTERS
%----------------------------------------------------------------------------------------

\RequirePackage[markcase=used]{scrlayer-scrpage}
\providepairofpagestyles{thesisSimple}{%
    \clearpairofpagestyles%
    \automark[chapter]{chapter}
    \ihead{\headmark}% Inner header
    \ohead[\pagemark]{\pagemark}% Outer header
}
\ifoot{}% Inner footer
\ofoot{}% Outer footer
\pagestyle{thesisSimple}
\providepairofpagestyles[thesisSimple]{thesis}{%
    \automark*[section]{}%
}
\providepairofpagestyles[thesisSimple]{review}{%
    \ofoot[\shorttitle/\authorname]{\shorttitle/\authorname}
    \ifoot[\tdate]{\tdate}
}
\pagestyle{thesis}
\ifbool{headsepline}{\KOMAoption{headsepline}{true}}{}
\PreventPackageFromLoading[\ClassError{\classname}{Package `fancyhdr' is
incompatible\MessageBreak with this class}{The pagesyles are defined 
    using package `scrlayer-scrpage', please consult the\MessageBreak
KOMA-script documentation for details.}]{fancyhdr}


\newcommand{\blank@p@gestyle}{empty}
\newcommand{\chapter@p@gestyle}{plain}
\NewDocumentCommand{\blankpagestyle}{ m }{%
    \ClassWarning{\classname}{\string\blankpagestyle\space is
    obsolete,\MessageBreak use \string\setblankpagestyle \space  instead}\renewcommand{\blank@p@gestyle}{}{#1}
}
\NewDocumentCommand{\setblankpagestyle}{ m }{\renewcommand{\blank@p@gestyle}{#1}}
\NewDocumentCommand{\setchapterpagestyle}{ m }{\renewcommand{\chapter@p@gestyle}{#1}}

\DeclareDocumentCommand\cleardoublepage{}{\clearpage\if@twoside \ifodd\c@page\else
    \hbox{}
    \thispagestyle{\blank@p@gestyle}
    \newpage
    \if@twocolumn\hbox{}\newpage\fi\fi\fi%
}

%----------------------------------------------------------------------------------------
%   ABBREVIATIONS PAGE DESIGN
%----------------------------------------------------------------------------------------

\newcommand{\abbrevname}{List of Abbreviations}
\providecaptionname{english,british,american}{\abbrevname}{List of Abbreviations}
\providecaptionname{ngerman,german,austrian,naustrian}{\abbrevname}{Abk\"urzungsverzeichnis}
\NewDocumentEnvironment{abbreviations}{ m }{%
    \ifbool{nolistspace}{\begingroup\singlespacing}{}
    \ifbool{listtoc}{\addchap{\abbrevname}}{\addchap*{\abbrevname}}
    \begin{longtable}{#1}
    }{%
    \end{longtable}
    \addtocounter{table}{-1}% Don't count this table as one of the document tables
    \ifbool{nolistspace}{\endgroup}{}
}

%----------------------------------------------------------------------------------------
%   ABSTRACT PAGE DESIGN
%----------------------------------------------------------------------------------------

\DeclareDocumentCommand{\abstractauthorfont}{}{}
\DeclareDocumentCommand{\abstracttitlefont}{}{}
\newcommand{\byname}{by}
\newcommand{\vonname}{von}
\newcommand{\abstractname}{Abstract}
\newcommand{\extraabstractname}{Zusammenfassung}
\providecaptionname{german,ngerman,austrian,naustrian}{\byname}{von}
\providecaptionname{american,australian,british,canadian,english,newzealand,UKenglish,USenglish}{\byname}{by}
\ifbool{consistentlayout}{
    \DeclareDocumentEnvironment{abstract}{ O{} }{%
        \addchap*{\abstractname}%
    }%
    {}%end alt-abstract
}{%
    \DeclareDocumentEnvironment{abstract}{ O{\null\vfill} }{
        \checktoopen
        \tttypeout{\abstractname}
        #1%added to be able to have abstract more than one page long
        \thispagestyle{plain}
        \begin{center}
            % Make this floating box inert to the parskip setting
            \setlength{\parskip}{0pt}
            {\normalsize \MakeUppercase{\univname}\par}% University name in capitals
            \bigskip
            {\Huge\textit{\abstractname} \par}
            \bigskip
            {\normalsize \instname \par}% Institute name
            {\normalsize \deptname \par}% Department name
            \bigskip
            {\normalsize \degreename\par}% Degree name
            \bigskip
            {\normalsize\bfseries \@title \par}% Thesis title
            \medskip
            {\normalsize \byname{} \authorname \par}% Author name
            \bigskip
        \end{center}
    }
    {
        \vfill\null
    }
}

\ifbool{consistentlayout}{
    \DeclareDocumentEnvironment{extraAbstract}{ O{} }{%
        \addchap*{\extraabstractname}%
    }%
    {}%end alt-abstract
}{%
    \DeclareDocumentEnvironment{extraAbstract}{ O{\null\vfill} }{
        \checktoopen
        \tttypeout{\extraabstractname}
        #1%added to be able to have abstract more than one page long
        \thispagestyle{empty}
        \begin{center}
            % Make this floating box inert to the parskip setting
            \setlength{\parskip}{0pt}
            {\normalsize \MakeUppercase{\univnameger} \par}% University name in capitals
            \bigskip
            {\Huge\textit{\extraabstractname} \par}
            \bigskip
            {\normalsize \instname \par}% Institute name
            {\normalsize \deptname \par}% Department name
            \bigskip
            {\normalsize \degreename\par}% Degree name
            \bigskip
            {\normalsize\bfseries \@title \par}% Thesis title
            \medskip
            {\normalsize \vonname{} \authorname \par}% Author name
            \bigskip
        \end{center}
    }
    {
        \vfill\null
    }
}

%----------------------------------------------------------------------------------------
%   ACKNOWLEDGEMENTS PAGE DESIGN
%----------------------------------------------------------------------------------------

\usepackage{xcolor}
\colorlet{mdtRed}{red!50!black}
\newcommand{\acknowledgementname}{Acknowledgements}
\providecaptionname{american,australian,british,canadian,english,newzealand,UKenglish,USenglish} {\acknowledgementname}{Acknowledgements} % Acknowledgement text for English countries
\providecaptionname{german,ngerman,austrian,naustrian}{\acknowledgementname}{Danksagung} % Acknowledgement text for Germanic countries

\ifbool{consistentlayout}{
    \DeclareDocumentEnvironment{acknowledgements}{}{%
        \tttypeout{\acknowledgementname}
        \addchap*{\acknowledgementname}
    }
}
{
    \DeclareDocumentEnvironment{acknowledgements}{}{%
        \checktoopen
        \tttypeout{\acknowledgementname}
        \thispagestyle{plain}
        \begin{center}{\Huge\textit{\acknowledgementname}\par}\end{center}
        \bigskip
    }
    {
        \vfill\null
    }
}


%----------------------------------------------------------------------------------------
%   DECLARATION PAGE DESIGN
%----------------------------------------------------------------------------------------

\newcommand{\authorshipname}{Declaration of Authorship}
\providecaptionname{american,australian,british,canadian,english,newzealand,UKenglish,USenglish}{\authorshipname}{Declaration of Authorship} % Declaration of Authorship text for English countries
\providecaptionname{german,ngerman,austrian,naustrian}{\authorshipname}{Eidesstattliche Erkl\"arung} % Declaration of Authorship text for Germanic countries

\ifbool{consistentlayout}{
    \DeclareDocumentEnvironment{declaration}{}{
        \addchap*{\authorshipname}
        }{}%
}{
    \DeclareDocumentEnvironment{declaration}{}{
        \checktoopen
        \tttypeout{\authorshipname}
        \thispagestyle{plain}
        \null\vfil
        \begin{center}{\Huge\textit{\authorshipname}\par}\end{center}
        \bigskip
    }{}
}


%----------------------------------------------------------------------------------------
%   DEDICATION PAGE DESIGN
%----------------------------------------------------------------------------------------
\newcommand\dedicatory[1]{
    \checktoopen
    \tttypeout{Dedicatory}
    \thispagestyle{empty}
    \null\vfil
    \begin{center}{\Large\slshape #1}\end{center}
    \vfil\null
}


%----------------------------------------------------------------------------------------
%   PHYSICAL CONSTANTS PAGE DESIGN
%----------------------------------------------------------------------------------------

\newcommand{\constantsname}{Physical Constants}
\providecaptionname{english,british,american}{\constantsname}{Physical Constants}
\providecaptionname{ngerman,german,austrian,naustrian}{\constantsname}{Physikalische Konstanten}

\NewDocumentEnvironment{constants}{ m }{%
    \ifbool{nolistspace}{\begingroup\singlespacing}{}
    \ifbool{listtoc}{\addchap{\constantsname}}{\addchap*{\constantsname}}
    \begin{longtable}{#1}
    }{%
    \end{longtable}
    \addtocounter{table}{-1}% Don't count this table as one of the document tables
    \ifbool{nolistspace}{\endgroup}{}
}

%----------------------------------------------------------------------------------------
%   SYMBOLS PAGE DESIGN
%----------------------------------------------------------------------------------------

\newcommand{\symbolsname}{List of Symbols}
\providecaptionname{english,british,american}{\symbolsname}{List of Symbols}
\providecaptionname{ngerman,german,austrian,naustrian}{\symbolsname}{Symbolverzeichnis}

\NewDocumentEnvironment{symbols}{ m }{%
    \ifbool{nolistspace}{\begingroup\singlespacing}{}
    \ifbool{listtoc}{\addchap{\symbolsname}}{\addchap*{\symbolsname}}
    \begin{longtable}{#1}
    }{%
    \end{longtable}
    \addtocounter{table}{-1}% Don't count this table as one of the document tables
    \ifbool{nolistspace}{\endgroup}{}
}

%----------------------------------------------------------------------------------------

\ifbool{hyperrefsupport}{% If the nohyperref class option has not been specified
\AtEndPreamble{\RequirePackage{hyperref}
\hypersetup{pdfpagemode={UseOutlines},
bookmarksopen=true,
bookmarksopenlevel=0,
hypertexnames=false,
colorlinks=true,% Set to false to disable coloring links
% citecolor=[rgb]{0.435, 0.031, 0},% The color of citations
% linkcolor=[rgb]{0.60, 0.043, 0},% The color of references to document elements (sections, figures, etc)
% urlcolor=[rgb]{0.733, 0.11, 0.063},% The color of hyperlinks (URLs)
citecolor=[rgb]{0.286, 0.427, 0.537},% The color of citations
linkcolor=[rgb]{0.161, 0.31, 0.427},% The color of references to document elements (sections, figures, etc)
urlcolor=[rgb]{0.071, 0.212, 0.322},% The color of hyperlinks (URLs)
pdfstartview={FitV},
unicode,
breaklinks=true,
}

\pdfstringdefDisableCommands{% If there is an explicit linebreak in a section heading (or anything printed to the pdf-bookmarks), it is replaced by a space
    \let\\\space%
}
    }
}{%nothing
}

%----------------------------------------------------------------------------------------

\endinput
% lazyLizardTracer
