import React, { useState } from 'react'
import {
    PaginationState,
    useReactTable,
    getCoreRowModel,
    getPaginationRowModel,
    ColumnDef,
    flexRender,
} from '@tanstack/react-table'

interface TableProps {
    data: any[];
    time: number | null;
}

export default function Table({ data, time }: TableProps) {
    const [pagination, setPagination] = useState<PaginationState>({
        pageIndex: 0,
        pageSize: 10,
    });

    const generateColumns = (): ColumnDef<any>[] => {
        if (data.length === 0) return [];

        const keys = Object.keys(data[0]);
        const dynamicColumns = keys.map((key) => ({
            accessorKey: key,
            header: () => key.charAt(0).toUpperCase() + key.slice(1),
            cell: (info: any) => info.getValue(),
            footer: (props: any) => props.column.id,
        }));

        return [
            {
                accessorKey: 'n',
                header: () => '#',
                cell: info => info.row.index + 1,
                footer: props => props.column.id,
            },
            ...dynamicColumns,
        ];
    };

    const columns = React.useMemo<ColumnDef<any>[]>(() => {
        return generateColumns();
    }, [data]);

    const table = useReactTable({
        data,
        columns,
        state: {
            pagination,
        },
        onPaginationChange: setPagination,
        getCoreRowModel: getCoreRowModel(),
        getPaginationRowModel: getPaginationRowModel(),
        debugTable: true,
    })

    return (
        <div className="w-full overflow-hidden">
            {data.length === 0 ? (
                <div className='flex justify-center'>
                    <p>...</p>
                </div>
            ) : (
                <div className='overflow-x-scroll'>
                    <table className="text-center table">
                        <thead>
                            {table.getHeaderGroups().map(headerGroup => (
                                <tr key={headerGroup.id}>
                                    {headerGroup.headers.map(header => {
                                        return (
                                            <th key={header.id} className={`px-2 border ${header.id == 'n' ? 'w-10' : ''}`}>
                                                {header.isPlaceholder ? null : (
                                                    <div>
                                                        {flexRender(
                                                            header.column.columnDef.header,
                                                            header.getContext()
                                                        )}
                                                    </div>
                                                )}
                                            </th>
                                        )
                                    })}
                                </tr>
                            ))}
                        </thead>
                        <tbody className='text-sm'>
                            {table.getRowModel().rows.map(row => {
                                return (
                                    <tr key={row.id}>
                                        {row.getVisibleCells().map(cell => {
                                            return (
                                                <td key={cell.id} className='border p-2 text-nowrap overflow-x-auto'>
                                                    {flexRender(
                                                        cell.column.columnDef.cell,
                                                        cell.getContext()
                                                    )}
                                                </td>
                                            )
                                        })}
                                    </tr>
                                )
                            })}
                        </tbody>
                    </table>
                </div>
            )}
            <div className='flex flex-col'>
                <div className='text-sm p-1 mb-4'>{data.length} Rows {time ? `in ${time.toFixed(2)} ms` : ""}</div>
                <div className="flex items-center gap-2">
                    <button
                        className={`border rounded p-1 bg-gray-50 ${table.getCanPreviousPage() ? 'hover:bg-gray-100' : ''}`}
                        onClick={() => table.setPageIndex(0)}
                        disabled={!table.getCanPreviousPage()}
                    >
                        {'<<'}
                    </button>
                    <button
                        className={`border rounded p-1 bg-gray-50 ${table.getCanPreviousPage() ? 'hover:bg-gray-100' : ''}`}
                        onClick={() => table.previousPage()}
                        disabled={!table.getCanPreviousPage()}
                    >
                        {'<'}
                    </button>
                    <button
                        className={`border rounded p-1 bg-gray-50 ${table.getCanPreviousPage() ? 'hover:bg-gray-100' : ''}`}
                        onClick={() => table.nextPage()}
                        disabled={!table.getCanNextPage()}
                    >
                        {'>'}
                    </button>
                    <button
                        className={`border rounded p-1 bg-gray-50 ${table.getCanPreviousPage() ? 'hover:bg-gray-100' : ''}`}
                        onClick={() => table.setPageIndex(table.getPageCount() - 1)}
                        disabled={!table.getCanNextPage()}
                    >
                        {'>>'}
                    </button>
                    <span className="flex items-center gap-1">
                        <div>Page</div>
                        <strong>
                            {table.getState().pagination.pageIndex + 1} of{' '}
                            {table.getPageCount()}
                        </strong>
                    </span>
                    <span className="flex items-center gap-1">
                        | Go to page:
                        <input
                            type="number"
                            min="1"
                            max={table.getPageCount()}
                            defaultValue={table.getState().pagination.pageIndex + 1}
                            onChange={e => {
                                const page = e.target.value ? Number(e.target.value) - 1 : 0
                                table.setPageIndex(page)
                            }}
                            className="border p-1 rounded w-16"
                        />
                    </span>
                    <select
                        value={table.getState().pagination.pageSize}
                        onChange={e => {
                            table.setPageSize(Number(e.target.value))
                        }}
                    >
                        {[10, 20, 30, 40, 50].map(pageSize => (
                            <option key={pageSize} value={pageSize}>
                                Show {pageSize}
                            </option>
                        ))}
                    </select>
                </div>
            </div>
        </div>
    )
}