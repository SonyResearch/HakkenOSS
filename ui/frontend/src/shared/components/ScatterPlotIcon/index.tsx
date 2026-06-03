export default function ScatterPlotIcon({ size = 64 }: { size: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <line x1="8" y1="56" x2="52" y2="56" stroke="#4B5563" strokeWidth="2.5" />
      <line x1="8" y1="56" x2="8" y2="12" stroke="#4B5563" strokeWidth="2.5" />

      <polygon points="56,56 52,53 52,59" fill="#4B5563" />
      <polygon points="8,8 5,12 11,12" fill="#4B5563" />

      <circle cx="20" cy="40" r="4" fill="#60b9f0" />
      <circle cx="28" cy="30" r="4" fill="#60b9f0" />
      <circle cx="36" cy="46" r="4" fill="#89c1e4" />
      <circle cx="44" cy="18" r="4" fill="#c8e3f4" />
      <circle cx="48" cy="34" r="4" fill="#60b9f0" />
    </svg>
  );
}
