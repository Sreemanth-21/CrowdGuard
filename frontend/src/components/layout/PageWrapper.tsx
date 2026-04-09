import { ReactNode } from 'react';

interface PageWrapperProps {
  children: ReactNode;
}

function PageWrapper({ children }: PageWrapperProps) {
  return (
    <main 
      className="ml-60 mt-14 min-h-[calc(100vh-3.5rem)] bg-navy-900"
      style={{ marginLeft: '240px', marginTop: '56px', minHeight: 'calc(100vh - 3.5rem)' }}
    >
      <div className="p-6" style={{ padding: '24px' }}>
        {children}
      </div>
    </main>
  );
}

export default PageWrapper;
