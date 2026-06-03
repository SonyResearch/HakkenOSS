interface TableOfContentsProps {
  contents: {
    title: string;
    id: string;
  }[];
}

const TableOfContents = ({ contents }: TableOfContentsProps) => {
  return (
    <div className="table-of-contents">
      <strong>Contents:</strong>
      <ul>
        {contents.map((content) => (
          <li key={content.id}>
            <a href={`#${content.id}`}>{content.title}</a>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default TableOfContents;
