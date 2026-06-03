/*Hook to determine if a user is using mobile at a given point*/

import { useEffect, useState } from 'react';

const query = '(max-width: 750px)';

export const useMobile = () => {
  const [isMobile, setIsMobile] = useState<boolean>(
    window.matchMedia(query).matches,
  );

  useEffect(() => {
    const media = window.matchMedia(query);

    const listener = (e: MediaQueryListEvent) => setIsMobile(e.matches);

    media.addEventListener('change', listener);

    return () => media.removeEventListener('change', listener);
  }, []);

  return isMobile;
};
