import { useEffect, useState } from 'react';
import { Reference } from '../../../../CandidateDashboard/types';
import { Button, Collapse, TableCell, TableRow } from '@mui/material';
import { highlight } from '../../ContextualizationSection/utils';
import { useMobile } from '../../../../../hooks/useMobile';
import { useLocation } from 'react-router-dom';

const ABSTRACT_REFS_URLS = {
  doi: 'https://doi.org/',
  pmid: 'https://pubmed.ncbi.nlm.nih.gov/',
  pmcid: 'https://www.ncbi.nlm.nih.gov/pmc/articles/',
};

export const ReferenceRow = ({
  reference,
  titleFilter,
  authorFilter,
  abstractFilter,
}: {
  reference: Reference;
  titleFilter: string;
  authorFilter: string;
  abstractFilter: string;
}) => {
  const location = useLocation();

  useEffect(() => {
    const cleanHash = location.hash.replace('#', '');

    const matching =
      `reference-${reference.publication_info.publication_id}` === cleanHash;

    setShowAbstract(matching);
  }, [location, reference.publication_info.publication_id]);
  const isMobile = useMobile();
  const [showAbstract, setShowAbstract] = useState<boolean>(false);
  const title = reference.publication_info.title;
  const authors = reference.publication_info.authors.map(
    (a) => `${a.first_name} ${a.last_name}`,
  );
  const abstract = reference.publication_info.abstract;
  const toggleButtonText = isMobile ? 'More' : 'Abstract';

  useEffect(() => {
    if (
      abstract &&
      abstractFilter &&
      abstract.toLowerCase().includes(abstractFilter.toLowerCase())
    ) {
      setShowAbstract(true);
    } else {
      setShowAbstract(false);
    }
  }, [abstractFilter, abstract]);

  const { doi, pmid, pmcid } = reference.publication_info;
  const urlIds = [
    doi && { name: 'doi', value: doi },
    pmid && { name: 'pmid', value: pmid },
    pmcid && { name: 'pmcid', value: pmcid },
  ].filter(Boolean) as { name: 'doi' | 'pmid' | 'pmcid'; value: string }[];
  return (
    <>
      <TableRow
        className="reference-row"
        id={`reference-${reference.publication_info.publication_id}`}
      >
        <TableCell
          dangerouslySetInnerHTML={{
            __html: highlight(title, titleFilter),
          }}
        ></TableCell>
        <TableCell sx={{ display: isMobile ? 'none' : 'table-cell' }}>
          {authors.map((author, index) => (
            <span
              key={index}
              dangerouslySetInnerHTML={{
                __html: highlight(`${author} `, authorFilter),
              }}
            ></span>
          ))}
        </TableCell>
        <TableCell sx={{ display: isMobile ? 'none' : 'table-cell' }}>
          {reference.publication_info.year}
        </TableCell>
        <TableCell sx={{ display: isMobile ? 'none' : 'table-cell' }}>
          {reference.publication_info.citations_count === 'None'
            ? 0
            : reference.publication_info.citations_count}
        </TableCell>
        <TableCell sx={{ display: isMobile ? 'none' : 'table-cell' }}>
          {Number(reference.score).toFixed(2)}
        </TableCell>
        {reference.publication_info.abstract ||
        reference.publication_info.doi ? (
          <TableCell align="right">
            <Button size="small" onClick={() => setShowAbstract(!showAbstract)}>
              {showAbstract
                ? `Hide ${toggleButtonText}`
                : `Show ${toggleButtonText}`}
            </Button>
          </TableCell>
        ) : (
          <TableCell></TableCell>
        )}
      </TableRow>
      <TableRow
        className="abstract-row"
        style={{ padding: showAbstract ? 'auto' : '0px', margin: '0px' }}
      >
        <TableCell colSpan={5} sx={{ padding: 0 }}>
          <Collapse sx={{ width: '100%' }} in={showAbstract} unmountOnExit>
            {isMobile && (
              <div className="extra-information">
                <p>
                  <strong>Year: </strong>
                  {reference.publication_info.year}
                </p>
                <p>
                  <strong>Citations count: </strong>
                  {reference.publication_info.citations_count}
                </p>
                <p>
                  <strong>Score: </strong>
                  {reference.score}
                </p>
                <p>
                  <strong>Authors: </strong>{' '}
                  {authors.map((author, index) => (
                    <span
                      key={index}
                      dangerouslySetInnerHTML={{
                        __html: highlight(`${author} `, authorFilter),
                      }}
                    ></span>
                  ))}
                </p>
              </div>
            )}
            {abstract && (
              <p
                dangerouslySetInnerHTML={{
                  __html: highlight(abstract, abstractFilter),
                }}
              ></p>
            )}
            <p>
              {urlIds.length > 0 &&
                urlIds.map((id, index) => (
                  <em key={index}>
                    <strong>{id.name.toUpperCase()}: </strong>
                    <a
                      target="_blank"
                      rel="noreferrer"
                      href={ABSTRACT_REFS_URLS[id.name] + id.value}
                    >
                      {id.value}{' '}
                    </a>
                  </em>
                ))}
            </p>
          </Collapse>
        </TableCell>
      </TableRow>
    </>
  );
};
