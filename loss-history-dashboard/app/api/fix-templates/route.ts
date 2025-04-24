import { NextResponse } from 'next/server';
import { prisma } from '@/lib/db';

// GET /api/fix-templates - Fetch all templates
export async function GET() {
  try {
    const templates = await prisma.fixTemplate.findMany({
      orderBy: {
        createdAt: 'desc'
      }
    });
    
    return NextResponse.json({ 
      success: true, 
      data: templates 
    });
  } catch (error) {
    console.error('Error fetching fix templates:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to fetch fix templates' },
      { status: 500 }
    );
  }
}

// POST /api/fix-templates - Create a new template
export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { name, description, problemType, solution, code } = body;
    
    // Basic validation
    if (!name || !problemType || !solution) {
      return NextResponse.json(
        { success: false, error: 'Missing required fields' },
        { status: 400 }
      );
    }
    
    const template = await prisma.fixTemplate.create({
      data: {
        name,
        description: description || '',
        problemType,
        solution,
        code: code || '',
      }
    });
    
    return NextResponse.json({
      success: true,
      data: template
    }, { status: 201 });
  } catch (error) {
    console.error('Error creating fix template:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to create fix template' },
      { status: 500 }
    );
  }
} 