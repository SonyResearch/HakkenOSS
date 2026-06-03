import './index.css';

interface PaginationProps {
  currentPage: number;
  numberOfPages: number;
  handleChangePage: (pageNumber: number) => void;
}
export const Pagination = ({
  currentPage,
  numberOfPages,
  handleChangePage,
}: PaginationProps) => {
  const getPagesToShow = () => {
    const pageNumbers = [];
    const pagesForward = 2;
    pageNumbers.push(1);
    if (currentPage - 1 > pagesForward) {
      pageNumbers.push('...'); //display last and first page w ... in between in case of many pages
    }
    for (let i = currentPage - pagesForward; i <= currentPage + 2; i++) {
      if (i > 1 && i < numberOfPages && currentPage - i < pagesForward) {
        pageNumbers.push(i);
      }
    }
    if (currentPage < numberOfPages - pagesForward - 1) {
      pageNumbers.push('...');
    }
    if (numberOfPages > 1) {
      pageNumbers.push(numberOfPages);
    }
    return pageNumbers;
  };
  const pageNumbers = getPagesToShow();

  return (
    <div className="pagination">
      <div>
        <button
          onClick={() => handleChangePage(currentPage - 1)}
          disabled={currentPage === numberOfPages - numberOfPages + 1}
        >
          Prev
        </button>
        {pageNumbers.map((pageNumber, index) => (
          <button
            onClick={() => handleChangePage(Number(pageNumber))}
            disabled={typeof pageNumber !== 'number'}
            className="pagination-number"
            key={index}
            style={
              pageNumber === currentPage
                ? {
                    color: 'var(--primary-pink)',
                    fontWeight: 'bold',
                    border: '1px solid var(--primary-pink)',
                  }
                : { fontWeight: 'normal' }
            }
          >
            {pageNumber}
          </button>
        ))}
        <button
          onClick={() => handleChangePage(currentPage + 1)}
          disabled={currentPage === numberOfPages}
        >
          Next
        </button>
      </div>
    </div>
  );
};
