STYLE = """
.custom-container {
    display: grid;
    align-items: center;
    margin: 0!important;
    overflow-y: hidden;
}
.prose ul ul {
    font-size: 10px!important;
}
.prose li {
    margin-bottom: 0!important;
}
.prose table {
    margin-bottom: 0!important;
    width: 100%;
    table-layout: fixed;
}
.prose td, th {
    padding: 2px 4px;
    text-wrap: normal;
    word-wrap: break-word;
    overflow-wrap: break-word;
    hyphens: auto;
}
.tree {
    padding: 0px;
    margin: 0!important;
    box-sizing: border-box;
    font-size: 10px;
    width: 100%;
    height: auto;
    text-align: center;
    display:inline-block;
    padding-bottom: 10px!important;
}
#root {
    display: inline-grid!important;
    width:auto!important;
    min-width: 220px;
}
.tree ul {
    padding-left: 20px;
    position: relative;
    transition: all 0.5s ease 0s;
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin: 0px !important;
}
.tree li {
    display: flex;
    text-align: center;
    list-style-type: none;
    position: relative;
    padding-left: 20px;
    transition: all 0.5s ease 0s;
    flex-direction: row;
    justify-content: start;
    align-items: center;
}
.tree li::before, .tree li::after {
    content: "";
    position: absolute;
    left: 0px;
    border-left: 1px solid var(--body-text-color);
    width: 20px;
}
.tree li::before {
    top: 0;
    height:50%;
}
.tree li::after {
    top: 50%;
    height: 55%;
    bottom: auto;
    border-top: 1px solid var(--body-text-color);
}
.tree li:only-child::after, li:only-child::before {
    display: none;
}
.tree li:first-child::before, .tree li:last-child::after {
    border: 0 none;
}
.tree li:last-child::before {
    border-bottom: 1px solid var(--body-text-color);
    border-radius: 0px 0px 0px 5px;
}
.tree li:first-child::after {
    border-radius: 5px 0 0 0;
}
.tree ul ul::before {
    content: "";
    position: absolute;
    left: 0;
    top: 50%;
    border-top: 1px solid var(--body-text-color);
    width: 20px;
    height: 0;
}
.tree ul:has(> li:only-child)::before {
    width:40px;
}
.tree li a {
    border: 1px solid var(--body-text-color);
    padding: 5px;
    text-decoration-line: none;
    border-radius: 5px;
    transition: .5s;
    display: flex;
    flex-direction: column;
    align-items: stretch;
    justify-content: space-between;
    overflow: hidden;
    width: 350px;  /* Increased width */
    max-width: 100%;
}
.tree li a span {
    padding: 5px;
    font-size: 12px;
    letter-spacing: 1px;
    font-weight: 500;
}
.tree li a:hover, .tree li a:hover+ul li a {
    background: var(--primary-500);
    color: white;
}
.tree li a:hover+ul li::after, .tree li a:hover+ul li::before,
.tree li a:hover+ul::before, .tree li a:hover+ul ul::before {
    border-color: var(--primary-500);
}
.end-of-text {
    width:auto!important;
    background-color: var(--secondary-500);
    color: white;
}
.nonfinal {
    background-color: white;
}
.chosen-step {
    background-color: var(--primary-500);
}
.chosen-step td {
    color: black!important;
}
"""
