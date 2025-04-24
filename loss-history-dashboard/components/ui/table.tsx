import React from 'react';

// Table
interface TableProps extends React.HTMLAttributes<HTMLTableElement> {
  children: React.ReactNode;
}

export const Table: React.FC<TableProps> = ({ children, className, ...props }) => {
  return (
    <div className="w-full overflow-auto">
      <table 
        className={`min-w-full divide-y divide-gray-200 ${className || ''}`} 
        {...props}
      >
        {children}
      </table>
    </div>
  );
};

// Table Header
interface TableHeaderProps extends React.HTMLAttributes<HTMLTableSectionElement> {
  children: React.ReactNode;
}

export const TableHeader: React.FC<TableHeaderProps> = ({ children, className, ...props }) => {
  return (
    <thead 
      className={`bg-gray-50 ${className || ''}`}
      {...props}
    >
      {children}
    </thead>
  );
};

// Table Body
interface TableBodyProps extends React.HTMLAttributes<HTMLTableSectionElement> {
  children: React.ReactNode;
}

export const TableBody: React.FC<TableBodyProps> = ({ children, className, ...props }) => {
  return (
    <tbody
      className={`bg-white divide-y divide-gray-200 ${className || ''}`}
      {...props}
    >
      {children}
    </tbody>
  );
};

// Table Row
interface TableRowProps extends React.HTMLAttributes<HTMLTableRowElement> {
  children: React.ReactNode;
}

export const TableRow: React.FC<TableRowProps> = ({ children, className, ...props }) => {
  return (
    <tr
      className={`hover:bg-gray-50 ${className || ''}`}
      {...props}
    >
      {children}
    </tr>
  );
};

// Table Head Cell
interface TableHeadCellProps extends React.ThHTMLAttributes<HTMLTableCellElement> {
  children: React.ReactNode;
}

export const TableHeadCell: React.FC<TableHeadCellProps> = ({ children, className, ...props }) => {
  return (
    <th
      scope="col"
      className={`px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider ${className || ''}`}
      {...props}
    >
      {children}
    </th>
  );
};

// Table Cell
interface TableCellProps extends React.TdHTMLAttributes<HTMLTableCellElement> {
  children: React.ReactNode;
}

export const TableCell: React.FC<TableCellProps> = ({ children, className, ...props }) => {
  return (
    <td
      className={`px-6 py-4 whitespace-nowrap text-sm text-gray-500 ${className || ''}`}
      {...props}
    >
      {children}
    </td>
  );
};
